from __future__ import annotations

import ipaddress
import logging
import subprocess
from collections.abc import Callable, Iterator
from typing import Any, ClassVar

import pytest

from tests.e2e.core.grpc_client import PRIVATE_API, PUBLIC_API, GRPCClient
from tests.e2e.core.helpers import (
    assert_grpc_rejected,
    unique_name,
    wait_for_cr,
    wait_for_deletion,
    wait_for_grpc_removal,
    wait_for_provision,
    wait_for_running,
    wait_for_security_group_cr,
    wait_for_security_group_deletion,
    wait_for_security_group_ready,
    wait_for_subnet_cr,
    wait_for_subnet_deletion,
    wait_for_subnet_ready,
    wait_for_virtual_network_cr,
    wait_for_virtual_network_deletion,
    wait_for_virtual_network_ready,
)
from tests.e2e.core.k8s_client import K8sClient
from tests.e2e.core.runner import poll_until
from tests.e2e.vmaas.regression import tenant_isolation_console as console

SOURCE_REF = "quay.io/containerdisks/fedora:41"

pytestmark = pytest.mark.regression

logger = logging.getLogger(__name__)

ALLOW_ALL_INGRESS: list[dict[str, str]] = [
    {"protocol": "PROTOCOL_ICMP", "ipv4_cidr": "0.0.0.0/0"},
    {"protocol": "PROTOCOL_ALL", "ipv4_cidr": "0.0.0.0/0"},
]
ALLOW_ALL_EGRESS: list[dict[str, str]] = [{"protocol": "PROTOCOL_ALL", "ipv4_cidr": "0.0.0.0/0"}]

_ADMIN_GET = {
    "vn": f"{PRIVATE_API}.VirtualNetworks/Get",
    "subnet": f"{PRIVATE_API}.Subnets/Get",
    "sg": f"{PRIVATE_API}.SecurityGroups/Get",
    "di": f"{PRIVATE_API}.DiskImages/Get",
    "ci": f"{PRIVATE_API}.ComputeInstances/Get",
}


def _require(state: dict[str, Any], *keys: str) -> None:
    missing = [k for k in keys if k not in state]
    if missing:
        pytest.skip(f"Prerequisite state missing: {', '.join(missing)}")


def _object(resp: dict[str, Any]) -> dict[str, Any]:
    obj = resp.get("object")
    return obj if isinstance(obj, dict) else resp


def _internal_ip(resp: dict[str, Any]) -> str:
    status = _object(resp).get("status") or {}
    value = status.get("internalIpAddress") or status.get("internal_ip_address") or ""
    return str(value)


def _spec(resp: dict[str, Any]) -> dict[str, Any]:
    spec = _object(resp).get("spec") or {}
    return spec if isinstance(spec, dict) else {}


def _rule_list(spec: dict[str, Any], *names: str) -> list[Any]:
    for name in names:
        if name in spec:
            value = spec[name]
            return value if isinstance(value, list) else []
    return []


def _attachment_sg_ids(resp: dict[str, Any]) -> list[str]:
    spec = _spec(resp)
    attachments = spec.get("networkAttachments") or spec.get("network_attachments") or []
    ids: list[str] = []
    if not isinstance(attachments, list):
        return ids
    for attachment in attachments:
        if not isinstance(attachment, dict):
            continue
        groups = attachment.get("securityGroups") or attachment.get("security_groups") or []
        if not isinstance(groups, list):
            continue
        for group in groups:
            if isinstance(group, dict) and group.get("id"):
                ids.append(str(group["id"]))
            elif isinstance(group, str):
                ids.append(group)
    return ids


def _tenant_cidrs(test_run_id: str) -> tuple[str, str]:
    third = int(test_run_id[:2], 16)
    if third in {0, 200}:
        third = 11
    return f"10.{third}.10.0/24", f"10.{third}.20.0/24"


def _assert_not_found(fn: Callable[[], object]) -> None:
    with pytest.raises(subprocess.CalledProcessError) as exc_info:
        fn()
    assert_grpc_rejected(exc_info, "NotFound")


def _assert_admin_get(private_grpc: GRPCClient, service: str, resource_id: str) -> None:
    resp = private_grpc.call(service=service, data={"id": resource_id})
    got = _object(resp).get("id")
    assert got == resource_id, f"admin Get {service} expected id {resource_id}, got {got!r}"


def _create_global_disk_image(private_grpc: GRPCClient, name: str) -> str:
    resp = private_grpc.call(
        service=f"{PRIVATE_API}.DiskImages/Create",
        data={
            "object": {
                "metadata": {"name": name},
                "spec": {
                    "source_type": "SOURCE_TYPE_REGISTRY",
                    "source_ref": SOURCE_REF,
                    "guest_os_family": "GUEST_OS_FAMILY_LINUX",
                    "architecture": ["ARCHITECTURE_AMD64"],
                },
            }
        },
    )
    return str(_object(resp)["id"])


def _provision_overlay(
    client: GRPCClient,
    k8s: K8sClient,
    *,
    prefix: str,
    cidr: str,
    state: dict[str, Any] | None = None,
    state_prefix: str = "",
) -> dict[str, str]:
    def _store(key: str, value: str) -> None:
        if state is not None:
            state[f"{state_prefix}_{key}"] = value

    vn_id = client.create_virtual_network(name=f"{prefix}-vn", ipv4_cidr=cidr)
    _store("vn_id", vn_id)
    vn_cr = wait_for_virtual_network_cr(k8s=k8s, uuid=vn_id)
    _store("vn_cr", vn_cr)
    wait_for_virtual_network_ready(k8s=k8s, name=vn_cr)
    subnet_id = client.create_subnet(name=f"{prefix}-sn", virtual_network=vn_id, ipv4_cidr=cidr)
    _store("subnet_id", subnet_id)
    subnet_cr = wait_for_subnet_cr(k8s=k8s, uuid=subnet_id)
    _store("subnet_cr", subnet_cr)
    wait_for_subnet_ready(k8s=k8s, name=subnet_cr)
    sg_id = client.create_security_group_with_rules(
        name=f"{prefix}-sg", virtual_network=vn_id, ingress=ALLOW_ALL_INGRESS, egress=ALLOW_ALL_EGRESS
    )
    _store("sg_id", sg_id)
    sg_cr = wait_for_security_group_cr(k8s=k8s, uuid=sg_id)
    _store("sg_cr", sg_cr)
    wait_for_security_group_ready(k8s=k8s, name=sg_cr)
    if state is not None:
        state[f"{state_prefix}_cidr"] = cidr
    return {
        "vn_id": vn_id,
        "vn_cr": vn_cr,
        "subnet_id": subnet_id,
        "subnet_cr": subnet_cr,
        "sg_id": sg_id,
        "sg_cr": sg_cr,
        "cidr": cidr,
    }


def _create_running_vm(
    client: GRPCClient,
    k8s: K8sClient,
    *,
    state: dict[str, Any],
    key: str,
    name: str,
    template: str,
    disk_image_name: str,
    subnet_id: str,
    sg_id: str,
    instance_type: str,
    storage_tier: str,
) -> dict[str, str]:
    ci_id = _create_vm(
        client,
        name=name,
        template=template,
        disk_image_name=disk_image_name,
        subnet_id=subnet_id,
        sg_id=sg_id,
        instance_type=instance_type,
        storage_tier=storage_tier,
    )
    state[key] = {"id": ci_id}
    vm = _wait_running_vm(client, k8s, ci_id)
    state[key] = vm
    return vm


def _create_vm(
    client: GRPCClient,
    *,
    name: str,
    template: str,
    disk_image_name: str,
    subnet_id: str,
    sg_id: str,
    instance_type: str,
    storage_tier: str,
) -> str:
    resp = client.call(
        service=f"{PUBLIC_API}.ComputeInstances/Create",
        data={
            "object": {
                "metadata": {"name": name},
                "spec": {
                    "template": {"name": template, "shared": True},
                    "disk_image": {"name": disk_image_name},
                    "instance_type": {"name": instance_type, "shared": True},
                    "boot_disk": {"storage_tier": {"name": storage_tier}},
                    "network_attachments": [{"subnet": {"id": subnet_id}, "security_groups": [{"id": sg_id}]}],
                    "user_data": console.user_data(),
                },
            }
        },
    )
    return str(_object(resp)["id"])


def _wait_running_vm(client: GRPCClient, k8s: K8sClient, ci_id: str) -> dict[str, str]:
    cr_name = wait_for_cr(k8s=k8s, uuid=ci_id)
    wait_for_provision(k8s=k8s, name=cr_name)
    wait_for_running(k8s=k8s, name=cr_name)
    ip = poll_until(
        fn=lambda: _internal_ip(client.get_compute_instance(ci_id=ci_id)),
        until=lambda value: value != "",
        retries=60,
        delay=5,
        description=f"ComputeInstance {ci_id} internal IP",
    )
    return {"id": ci_id, "cr": cr_name, "ip": ip}


def _ip_in_cidr(ip: str, cidr: str) -> bool:
    return ipaddress.ip_address(ip) in ipaddress.ip_network(cidr)


def _guest_ping(client: GRPCClient, fulfillment_address: str, vm_id: str, dest_ip: str) -> bool:
    return console.ping(grpc=client, fulfillment_address=fulfillment_address, vm_id=vm_id, dest_ip=dest_ip)


def _guest_ping_retrying(client: GRPCClient, fulfillment_address: str, vm_id: str, dest_ip: str) -> bool:
    """Treat console/login glitches as not-yet-ready so success polls can retry."""
    try:
        return _guest_ping(client, fulfillment_address, vm_id, dest_ip)
    except AssertionError as exc:
        logger.warning("Guest ping not yet observable: %s", exc)
        return False


def _assert_ping_succeeds(
    client: GRPCClient, fulfillment_address: str, vm_id: str, dest_ip: str, description: str
) -> None:
    poll_until(
        fn=lambda: _guest_ping_retrying(client, fulfillment_address, vm_id, dest_ip),
        until=lambda ok: ok,
        retries=12,
        delay=10,
        description=description,
    )


def _assert_ping_fails(
    client: GRPCClient, fulfillment_address: str, vm_id: str, dest_ip: str, description: str
) -> None:
    for attempt in range(3):
        try:
            ok = _guest_ping(client, fulfillment_address, vm_id, dest_ip)
        except AssertionError as exc:
            pytest.fail(f"{description} did not observe ICMP (attempt {attempt + 1}): {exc}")
        if ok:
            pytest.fail(f"{description} unexpectedly succeeded (attempt {attempt + 1})")


def _assert_ping_stops(
    client: GRPCClient, fulfillment_address: str, vm_id: str, dest_ip: str, description: str
) -> None:
    poll_until(
        fn=lambda: _guest_ping(client, fulfillment_address, vm_id, dest_ip),
        until=lambda ok: not ok,
        retries=8,
        delay=10,
        description=description,
    )


def _best_effort_delete_vm(client: GRPCClient, k8s: K8sClient, vm: dict[str, str] | None) -> None:
    if not vm or not vm.get("id"):
        return
    try:
        client.delete_compute_instance(ci_id=vm["id"])
        if vm.get("cr"):
            wait_for_deletion(k8s=k8s, name=vm["cr"])
        wait_for_grpc_removal(grpc=client, uuid=vm["id"])
    except Exception as exc:
        logger.warning("Failed to delete ComputeInstance %s: %s", vm.get("id"), exc)


def _best_effort_delete_overlay(client: GRPCClient, k8s: K8sClient, prefix: str, state: dict[str, Any]) -> None:
    sg_id, sg_cr = state.get(f"{prefix}_sg_id"), state.get(f"{prefix}_sg_cr")
    subnet_id, subnet_cr = state.get(f"{prefix}_subnet_id"), state.get(f"{prefix}_subnet_cr")
    vn_id, vn_cr = state.get(f"{prefix}_vn_id"), state.get(f"{prefix}_vn_cr")
    if sg_id:
        try:
            client.delete_security_group(sg_id=sg_id)
            if sg_cr:
                wait_for_security_group_deletion(k8s=k8s, name=sg_cr)
        except Exception as exc:
            logger.warning("Failed to delete SecurityGroup %s: %s", sg_id, exc)
    if subnet_id:
        try:
            client.delete_subnet(subnet_id=subnet_id)
            if subnet_cr:
                wait_for_subnet_deletion(k8s=k8s, name=subnet_cr)
        except Exception as exc:
            logger.warning("Failed to delete Subnet %s: %s", subnet_id, exc)
    if vn_id:
        try:
            client.delete_virtual_network(vn_id=vn_id)
            if vn_cr:
                wait_for_virtual_network_deletion(k8s=k8s, name=vn_cr)
        except Exception as exc:
            logger.warning("Failed to delete VirtualNetwork %s: %s", vn_id, exc)


def _best_effort_delete_disk_image(client: GRPCClient, disk_image_id: str | None) -> None:
    if not disk_image_id:
        return
    try:
        client.delete_disk_image(disk_image_id=disk_image_id)
    except Exception as exc:
        logger.warning("Failed to delete DiskImage %s: %s", disk_image_id, exc)


def _cleanup_resources(
    state: dict[str, Any],
    jwt_grpc_tenant1: GRPCClient,
    jwt_grpc_tenant2: GRPCClient,
    private_grpc: GRPCClient,
    k8s: K8sClient,
) -> None:
    _best_effort_delete_vm(jwt_grpc_tenant1, k8s, state.get("t1_vm1"))
    _best_effort_delete_vm(jwt_grpc_tenant1, k8s, state.get("t1_vm2"))
    _best_effort_delete_vm(jwt_grpc_tenant2, k8s, state.get("t2_vm1"))
    _best_effort_delete_vm(jwt_grpc_tenant2, k8s, state.get("t2_vm2"))
    _best_effort_delete_vm(jwt_grpc_tenant2, k8s, state.get("copycat_vm"))
    _best_effort_delete_overlay(jwt_grpc_tenant2, k8s, "copycat", state)
    _best_effort_delete_overlay(jwt_grpc_tenant1, k8s, "t1", state)
    _best_effort_delete_overlay(jwt_grpc_tenant2, k8s, "t2", state)
    _best_effort_delete_disk_image(jwt_grpc_tenant1, state.get("t1_di_id"))
    _best_effort_delete_disk_image(jwt_grpc_tenant2, state.get("t2_di_id"))
    global_di_id = state.get("global_di_id")
    if global_di_id:
        try:
            private_grpc.call(service=f"{PRIVATE_API}.DiskImages/Delete", data={"id": global_di_id})
        except Exception as exc:
            logger.warning("Failed to delete global DiskImage %s: %s", global_di_id, exc)


@pytest.fixture(scope="class", autouse=True)
def _tenant_isolation_cleanup(
    request: pytest.FixtureRequest,
    jwt_grpc_tenant1: GRPCClient,
    jwt_grpc_tenant2: GRPCClient,
    private_grpc: GRPCClient,
    k8s_hub_client: K8sClient,
) -> Iterator[None]:
    yield
    cls = request.cls
    if cls is None:
        return
    _cleanup_resources(cls.state, jwt_grpc_tenant1, jwt_grpc_tenant2, private_grpc, k8s_hub_client)


class TestVmaasTenantIsolation:
    state: ClassVar[dict[str, Any]] = {}

    def test_01_overlay_and_catalog_isolation(
        self,
        private_grpc: GRPCClient,
        jwt_grpc_tenant1: GRPCClient,
        jwt_grpc_tenant2: GRPCClient,
        k8s_hub_client: K8sClient,
        test_run_id: str,
    ) -> None:
        self.__class__.state.clear()
        state = self.__class__.state
        t1_cidr, t2_cidr = _tenant_cidrs(test_run_id)
        state["t1_cidr"] = t1_cidr
        state["t2_cidr"] = t2_cidr
        global_di_id = _create_global_disk_image(private_grpc, unique_name(f"iso-global-{test_run_id}"))
        state["global_di_id"] = global_di_id
        t1_overlay = _provision_overlay(
            jwt_grpc_tenant1,
            k8s_hub_client,
            prefix=f"iso-t1-{test_run_id}",
            cidr=t1_cidr,
            state=state,
            state_prefix="t1",
        )
        t2_overlay = _provision_overlay(
            jwt_grpc_tenant2,
            k8s_hub_client,
            prefix=f"iso-t2-{test_run_id}",
            cidr=t2_cidr,
            state=state,
            state_prefix="t2",
        )
        t1_di_name = unique_name(f"iso-t1-di-{test_run_id}")
        t2_di_name = unique_name(f"iso-t2-di-{test_run_id}")
        t1_di_id = jwt_grpc_tenant1.create_disk_image(name=t1_di_name, source_ref=SOURCE_REF)
        state.update(t1_di_id=t1_di_id, t1_di_name=t1_di_name)
        t2_di_id = jwt_grpc_tenant2.create_disk_image(name=t2_di_name, source_ref=SOURCE_REF)
        state.update(t2_di_id=t2_di_id, t2_di_name=t2_di_name)

        for service, resource_id in (
            (_ADMIN_GET["vn"], t1_overlay["vn_id"]),
            (_ADMIN_GET["vn"], t2_overlay["vn_id"]),
            (_ADMIN_GET["subnet"], t1_overlay["subnet_id"]),
            (_ADMIN_GET["subnet"], t2_overlay["subnet_id"]),
            (_ADMIN_GET["sg"], t1_overlay["sg_id"]),
            (_ADMIN_GET["sg"], t2_overlay["sg_id"]),
            (_ADMIN_GET["di"], t1_di_id),
            (_ADMIN_GET["di"], t2_di_id),
            (_ADMIN_GET["di"], global_di_id),
        ):
            _assert_admin_get(private_grpc, service, resource_id)

        t1_images = jwt_grpc_tenant1.list_disk_image_ids()
        t2_images = jwt_grpc_tenant2.list_disk_image_ids()
        assert t1_di_id in t1_images
        assert global_di_id in t1_images
        assert t2_di_id not in t1_images
        assert t2_di_id in t2_images
        assert global_di_id in t2_images
        assert t1_di_id not in t2_images
        _assert_not_found(lambda: jwt_grpc_tenant2.get_disk_image(disk_image_id=t1_di_id))

        assert t1_overlay["vn_id"] in jwt_grpc_tenant1.list_virtual_network_ids()
        assert t1_overlay["subnet_id"] in jwt_grpc_tenant1.list_subnet_ids()
        assert t1_overlay["sg_id"] in jwt_grpc_tenant1.list_security_group_ids()
        assert t2_overlay["vn_id"] not in jwt_grpc_tenant1.list_virtual_network_ids()
        assert t2_overlay["subnet_id"] not in jwt_grpc_tenant1.list_subnet_ids()
        assert t2_overlay["sg_id"] not in jwt_grpc_tenant1.list_security_group_ids()
        _assert_not_found(lambda: jwt_grpc_tenant1.get_virtual_network(vn_id=t2_overlay["vn_id"]))
        _assert_not_found(lambda: jwt_grpc_tenant1.get_subnet(subnet_id=t2_overlay["subnet_id"]))
        _assert_not_found(lambda: jwt_grpc_tenant1.get_security_group(sg_id=t2_overlay["sg_id"]))

        assert t2_overlay["vn_id"] in jwt_grpc_tenant2.list_virtual_network_ids()
        assert t2_overlay["subnet_id"] in jwt_grpc_tenant2.list_subnet_ids()
        assert t2_overlay["sg_id"] in jwt_grpc_tenant2.list_security_group_ids()
        assert t1_overlay["vn_id"] not in jwt_grpc_tenant2.list_virtual_network_ids()
        assert t1_overlay["subnet_id"] not in jwt_grpc_tenant2.list_subnet_ids()
        assert t1_overlay["sg_id"] not in jwt_grpc_tenant2.list_security_group_ids()
        _assert_not_found(lambda: jwt_grpc_tenant2.get_virtual_network(vn_id=t1_overlay["vn_id"]))
        _assert_not_found(lambda: jwt_grpc_tenant2.get_subnet(subnet_id=t1_overlay["subnet_id"]))
        _assert_not_found(lambda: jwt_grpc_tenant2.get_security_group(sg_id=t1_overlay["sg_id"]))

    def test_02_tenants_create_vms(
        self,
        private_grpc: GRPCClient,
        jwt_grpc_tenant1: GRPCClient,
        jwt_grpc_tenant2: GRPCClient,
        k8s_hub_client: K8sClient,
        vm_template: str,
        default_instance_type: str,
        default_storage_tier: str,
        test_run_id: str,
    ) -> None:
        _require(self.state, "t1_subnet_id", "t1_sg_id", "t1_di_name", "t2_subnet_id", "t2_sg_id", "t2_di_name")
        state = self.__class__.state

        t1_vm1 = _create_running_vm(
            jwt_grpc_tenant1,
            k8s_hub_client,
            state=state,
            key="t1_vm1",
            name=unique_name(f"iso-t1-vm1-{test_run_id}"),
            template=vm_template,
            disk_image_name=state["t1_di_name"],
            subnet_id=state["t1_subnet_id"],
            sg_id=state["t1_sg_id"],
            instance_type=default_instance_type,
            storage_tier=default_storage_tier,
        )
        t1_vm2 = _create_running_vm(
            jwt_grpc_tenant1,
            k8s_hub_client,
            state=state,
            key="t1_vm2",
            name=unique_name(f"iso-t1-vm2-{test_run_id}"),
            template=vm_template,
            disk_image_name=state["t1_di_name"],
            subnet_id=state["t1_subnet_id"],
            sg_id=state["t1_sg_id"],
            instance_type=default_instance_type,
            storage_tier=default_storage_tier,
        )
        t2_vm1 = _create_running_vm(
            jwt_grpc_tenant2,
            k8s_hub_client,
            state=state,
            key="t2_vm1",
            name=unique_name(f"iso-t2-vm1-{test_run_id}"),
            template=vm_template,
            disk_image_name=state["t2_di_name"],
            subnet_id=state["t2_subnet_id"],
            sg_id=state["t2_sg_id"],
            instance_type=default_instance_type,
            storage_tier=default_storage_tier,
        )
        t2_vm2 = _create_running_vm(
            jwt_grpc_tenant2,
            k8s_hub_client,
            state=state,
            key="t2_vm2",
            name=unique_name(f"iso-t2-vm2-{test_run_id}"),
            template=vm_template,
            disk_image_name=state["t2_di_name"],
            subnet_id=state["t2_subnet_id"],
            sg_id=state["t2_sg_id"],
            instance_type=default_instance_type,
            storage_tier=default_storage_tier,
        )

        assert _ip_in_cidr(t1_vm1["ip"], state["t1_cidr"])
        assert _ip_in_cidr(t1_vm2["ip"], self.state["t1_cidr"])
        assert _ip_in_cidr(t2_vm1["ip"], self.state["t2_cidr"])
        assert _ip_in_cidr(t2_vm2["ip"], self.state["t2_cidr"])

        t1_ids = jwt_grpc_tenant1.list_compute_instance_ids()
        t2_ids = jwt_grpc_tenant2.list_compute_instance_ids()
        assert t1_vm1["id"] in t1_ids and t1_vm2["id"] in t1_ids
        assert t2_vm1["id"] not in t1_ids and t2_vm2["id"] not in t1_ids
        assert t2_vm1["id"] in t2_ids and t2_vm2["id"] in t2_ids
        assert t1_vm1["id"] not in t2_ids and t1_vm2["id"] not in t2_ids

        _assert_not_found(lambda: jwt_grpc_tenant1.get_compute_instance(ci_id=t2_vm1["id"]))
        _assert_not_found(lambda: jwt_grpc_tenant2.get_compute_instance(ci_id=t1_vm1["id"]))

        for vm in (t1_vm1, t1_vm2, t2_vm1, t2_vm2):
            _assert_admin_get(private_grpc, _ADMIN_GET["ci"], vm["id"])

        t1_ci = jwt_grpc_tenant1.get_compute_instance(ci_id=t1_vm1["id"])
        assert self.state["t1_sg_id"] in _attachment_sg_ids(t1_ci)

    def test_03_intra_tenant_ping(
        self, jwt_grpc_tenant1: GRPCClient, jwt_grpc_tenant2: GRPCClient, fulfillment_address: str
    ) -> None:
        _require(self.state, "t1_vm1", "t1_vm2", "t2_vm1", "t2_vm2")
        _assert_ping_succeeds(
            jwt_grpc_tenant1,
            fulfillment_address,
            self.state["t1_vm1"]["id"],
            self.state["t1_vm2"]["ip"],
            "Tenant-1 intra-overlay ICMP",
        )
        _assert_ping_succeeds(
            jwt_grpc_tenant2,
            fulfillment_address,
            self.state["t2_vm1"]["id"],
            self.state["t2_vm2"]["ip"],
            "Tenant-2 intra-overlay ICMP",
        )

    def test_04_cross_tenant_and_copycat_ping_fails(
        self,
        jwt_grpc_tenant1: GRPCClient,
        jwt_grpc_tenant2: GRPCClient,
        k8s_hub_client: K8sClient,
        fulfillment_address: str,
        vm_template: str,
        default_instance_type: str,
        default_storage_tier: str,
        test_run_id: str,
    ) -> None:
        _require(self.state, "t1_vm1", "t2_vm1", "t1_cidr", "t2_di_name")
        _assert_ping_fails(
            jwt_grpc_tenant1,
            fulfillment_address,
            self.state["t1_vm1"]["id"],
            self.state["t2_vm1"]["ip"],
            "Cross-tenant ICMP",
        )

        copycat = _provision_overlay(
            jwt_grpc_tenant2,
            k8s_hub_client,
            prefix=f"iso-copycat-{test_run_id}",
            cidr=self.state["t1_cidr"],
            state=self.__class__.state,
            state_prefix="copycat",
        )
        copycat_vm = _create_running_vm(
            jwt_grpc_tenant2,
            k8s_hub_client,
            state=self.__class__.state,
            key="copycat_vm",
            name=unique_name(f"iso-copycat-{test_run_id}"),
            template=vm_template,
            disk_image_name=self.state["t2_di_name"],
            subnet_id=copycat["subnet_id"],
            sg_id=copycat["sg_id"],
            instance_type=default_instance_type,
            storage_tier=default_storage_tier,
        )
        assert _ip_in_cidr(copycat_vm["ip"], self.state["t1_cidr"])
        assert copycat_vm["ip"] not in {self.state["t1_vm1"]["ip"], self.state["t1_vm2"]["ip"]}
        _assert_ping_fails(
            jwt_grpc_tenant1,
            fulfillment_address,
            self.state["t1_vm1"]["id"],
            copycat_vm["ip"],
            "Copycat same-CIDR ICMP",
        )

    def test_05_live_security_group_stateful(self, jwt_grpc_tenant1: GRPCClient, fulfillment_address: str) -> None:
        _require(self.state, "t1_sg_id", "t1_vm1", "t1_vm2")
        sg_id = self.state["t1_sg_id"]
        vm1 = self.state["t1_vm1"]
        dest_ip = self.state["t1_vm2"]["ip"]

        jwt_grpc_tenant1.update_security_group_rules(sg_id=sg_id, ingress=[])
        sg = jwt_grpc_tenant1.get_security_group(sg_id=sg_id)
        assert _rule_list(_spec(sg), "ingress") == []
        assert _rule_list(_spec(sg), "egress"), "ingress-only update must leave egress rules"
        _assert_ping_succeeds(
            jwt_grpc_tenant1, fulfillment_address, vm1["id"], dest_ip, "ICMP after ingress-only SecurityGroup clear"
        )

        jwt_grpc_tenant1.update_security_group_rules(sg_id=sg_id, ingress=[], egress=[])
        sg = jwt_grpc_tenant1.get_security_group(sg_id=sg_id)
        assert _rule_list(_spec(sg), "ingress") == []
        assert _rule_list(_spec(sg), "egress") == []
        _assert_ping_stops(
            jwt_grpc_tenant1,
            fulfillment_address,
            vm1["id"],
            dest_ip,
            "ICMP after ingress and egress SecurityGroup clear",
        )

        jwt_grpc_tenant1.update_security_group_rules(sg_id=sg_id, ingress=ALLOW_ALL_INGRESS, egress=ALLOW_ALL_EGRESS)
        _assert_ping_succeeds(
            jwt_grpc_tenant1, fulfillment_address, vm1["id"], dest_ip, "ICMP after SecurityGroup restore"
        )

    def test_06_tenant2_deletes_vms(self, jwt_grpc_tenant2: GRPCClient, k8s_hub_client: K8sClient) -> None:
        _require(self.state, "t2_vm1", "t2_vm2")
        deleted: list[str] = []
        for key in ("t2_vm1", "t2_vm2", "copycat_vm"):
            vm = self.state.get(key)
            if not vm:
                continue
            jwt_grpc_tenant2.delete_compute_instance(ci_id=vm["id"])
            if vm.get("cr"):
                wait_for_deletion(k8s=k8s_hub_client, name=vm["cr"])
            wait_for_grpc_removal(grpc=jwt_grpc_tenant2, uuid=vm["id"])
            deleted.append(vm["id"])
            self.state.pop(key, None)
        remaining = jwt_grpc_tenant2.list_compute_instance_ids()
        for ci_id in deleted:
            assert ci_id not in remaining

    def test_07_tenant1_unaffected(
        self, jwt_grpc_tenant1: GRPCClient, k8s_hub_client: K8sClient, fulfillment_address: str
    ) -> None:
        _require(self.state, "t1_vm1", "t1_vm2")
        for vm in (self.state["t1_vm1"], self.state["t1_vm2"]):
            phase = k8s_hub_client.get_compute_instance_phase(name=vm["cr"], checked=False)
            assert phase == "Running", f"{vm['id']} expected Running, got {phase!r}"
            jwt_grpc_tenant1.get_compute_instance(ci_id=vm["id"])
        _assert_ping_succeeds(
            jwt_grpc_tenant1,
            fulfillment_address,
            self.state["t1_vm1"]["id"],
            self.state["t1_vm2"]["ip"],
            "Tenant-1 ICMP after Tenant-2 delete",
        )
