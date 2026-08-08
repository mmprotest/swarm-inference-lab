"""Builders for canonical decoder component contracts."""

from __future__ import annotations

from swarm_inference.engines.interfaces import (
    ComponentBoundaryContract,
    ComponentPlacement,
    ExecutionComponent,
    ExecutionComponentType,
)


def _contract(
    boundary_id: str,
    value_kind: str,
    shape: tuple[int | str, ...],
    dtype: str,
    device: str,
    revision: str,
    *,
    batch_dimension: int | None,
    sequence_dimension: int | None,
    sequence_position: str = "absolute-cache-position",
    token_position: str = "same-as-input",
    kv_identity: str | None = None,
    route_identity: str | None = None,
) -> ComponentBoundaryContract:
    return ComponentBoundaryContract(
        boundary_id=boundary_id,
        value_kind=value_kind,  # type: ignore[arg-type]
        shape=shape,
        dtype=dtype,
        device=device,
        model_revision=revision,
        batch_dimension=batch_dimension,
        sequence_dimension=sequence_dimension,
        sequence_position=sequence_position,  # type: ignore[arg-type]
        token_position=token_position,  # type: ignore[arg-type]
        kv_identity=kv_identity,
        route_identity=route_identity,
    )


def sparse_outer_components(
    *,
    engine_id: str,
    architecture_id: str,
    revision: str,
    worker_id: str,
    device: str,
    hidden_size: int,
    layer_count: int,
    experts_per_token: int,
    dtype: str,
    required_memory_bytes: int,
) -> tuple[ExecutionComponent, ...]:
    """Native decoder work surrounding an external routed-expert component."""

    cpu = ComponentPlacement(worker_ids=(worker_id,), device="cpu", persistent=True)
    compute = ComponentPlacement(
        worker_ids=(worker_id,),
        device=device,
        memory_tiers=("vram", "ram"),
        persistent=True,
        direct_data_path=True,
        bounded_queue_depth=2,
        colocation_group="decoder-compute",
        require_same_device=True,
    )
    hidden = ("batch", "sequence", hidden_size)
    token_ids = ("batch", "sequence")
    tokens_out = _contract(
        "tokenizer-token-ids",
        "token-ids",
        token_ids,
        "int64",
        "cpu",
        revision,
        batch_dimension=0,
        sequence_dimension=1,
        sequence_position="prompt-relative",
    )
    embedding_in = tokens_out.model_copy(
        update={"boundary_id": "embedding-token-ids", "device": device}
    )
    embedding_out = _contract(
        "embedded-hidden",
        "hidden-states",
        hidden,
        dtype,
        device,
        revision,
        batch_dimension=0,
        sequence_dimension=1,
    )
    attention_in = embedding_out.model_copy(update={"boundary_id": "attention-hidden-in"})
    attention_out = embedding_out.model_copy(update={"boundary_id": "attention-hidden-out"})
    kv_state = _contract(
        "attention-kv-state",
        "kv-state",
        (layer_count, "batch", "kv-heads", "sequence", "head-dim"),
        dtype,
        device,
        revision,
        batch_dimension=1,
        sequence_dimension=3,
        kv_identity=f"{architecture_id}@{revision}:stage-kv",
    )
    router_in = attention_out.model_copy(update={"boundary_id": "router-hidden-in"})
    routes = _contract(
        "router-expert-routes",
        "expert-routes",
        ("batch", "sequence", experts_per_token),
        "adapter-exact-route",
        device,
        revision,
        batch_dimension=0,
        sequence_dimension=1,
        route_identity=f"{architecture_id}@{revision}:route",
    )
    routed_out = attention_out.model_copy(update={"boundary_id": "routed-hidden-out"})
    norm_in = routed_out.model_copy(update={"boundary_id": "norm-hidden-in"})
    norm_out = routed_out.model_copy(update={"boundary_id": "norm-hidden-out"})
    head_in = norm_out.model_copy(update={"boundary_id": "head-hidden-in"})
    logits = _contract(
        "head-logits",
        "logits",
        ("batch", "sequence", "vocabulary"),
        "float32",
        device,
        revision,
        batch_dimension=0,
        sequence_dimension=1,
    )
    sampling_in = logits.model_copy(update={"boundary_id": "sampling-logits"})
    sampled = _contract(
        "sampled-token-ids",
        "sampled-token-ids",
        ("batch",),
        "int64",
        device,
        revision,
        batch_dimension=0,
        sequence_dimension=None,
        sequence_position="current-token",
        token_position="next-token",
    )
    publication_in = sampled.model_copy(
        update={"boundary_id": "publication-token-ids", "device": "cpu"}
    )
    published = _contract(
        "published-token",
        "published-token",
        ("batch",),
        "utf8+int64",
        "cpu",
        revision,
        batch_dimension=0,
        sequence_dimension=None,
        sequence_position="current-token",
        token_position="next-token",
    )
    shared = {
        "engine_id": engine_id,
        "architecture_id": architecture_id,
        "model_revision": revision,
    }
    per_component_memory = max(1, required_memory_bytes // 7)
    return (
        ExecutionComponent(
            component_id="tokenization",
            component_type=ExecutionComponentType.TOKENIZATION,
            placement=cpu,
            output_contracts=(tokens_out,),
            estimated_memory_bytes=64 * 1024 * 1024,
            **shared,
        ),
        ExecutionComponent(
            component_id="embedding",
            component_type=ExecutionComponentType.EMBEDDING,
            placement=compute,
            input_contracts=(embedding_in,),
            output_contracts=(embedding_out,),
            depends_on=("tokenization",),
            estimated_memory_bytes=per_component_memory,
            **shared,
        ),
        ExecutionComponent(
            component_id="attention",
            component_type=ExecutionComponentType.ATTENTION,
            placement=compute,
            input_contracts=(attention_in,),
            output_contracts=(attention_out, kv_state),
            depends_on=("embedding",),
            estimated_memory_bytes=per_component_memory,
            **shared,
        ),
        ExecutionComponent(
            component_id="kv-cache",
            component_type=ExecutionComponentType.KV_CACHE,
            placement=compute,
            input_contracts=(kv_state.model_copy(update={"boundary_id": "kv-cache-state-in"}),),
            depends_on=("attention",),
            estimated_memory_bytes=per_component_memory,
            metadata={"identity_isolation": "request+sequence+position+generation"},
            **shared,
        ),
        ExecutionComponent(
            component_id="router",
            component_type=ExecutionComponentType.ROUTER,
            placement=compute,
            input_contracts=(router_in,),
            output_contracts=(routes,),
            depends_on=("attention",),
            estimated_memory_bytes=per_component_memory,
            **shared,
        ),
        ExecutionComponent(
            component_id="normalization",
            component_type=ExecutionComponentType.NORMALIZATION,
            placement=compute,
            input_contracts=(norm_in,),
            output_contracts=(norm_out,),
            depends_on=("routed-experts",),
            estimated_memory_bytes=per_component_memory,
            **shared,
        ),
        ExecutionComponent(
            component_id="lm-head",
            component_type=ExecutionComponentType.LM_HEAD,
            placement=compute,
            input_contracts=(head_in,),
            output_contracts=(logits,),
            depends_on=("normalization",),
            estimated_memory_bytes=per_component_memory,
            **shared,
        ),
        ExecutionComponent(
            component_id="sampling",
            component_type=ExecutionComponentType.SAMPLING,
            placement=compute,
            input_contracts=(sampling_in,),
            output_contracts=(sampled,),
            depends_on=("lm-head",),
            estimated_memory_bytes=16 * 1024 * 1024,
            **shared,
        ),
        ExecutionComponent(
            component_id="token-publication",
            component_type=ExecutionComponentType.TOKEN_PUBLICATION,
            placement=cpu,
            input_contracts=(publication_in,),
            output_contracts=(published,),
            depends_on=("sampling",),
            estimated_memory_bytes=16 * 1024 * 1024,
            **shared,
        ),
    )


def routed_expert_component(
    *,
    engine_id: str,
    architecture_id: str,
    revision: str,
    worker_ids: tuple[str, ...],
    device: str,
    hidden_size: int,
    experts_per_token: int,
    dtype: str,
    required_memory_bytes: int,
    metadata: dict[str, object],
) -> ExecutionComponent:
    placement = ComponentPlacement(
        worker_ids=worker_ids,
        device=device,
        memory_tiers=("vram", "ram", "nvme"),
        persistent=True,
        direct_data_path=True,
        bounded_queue_depth=4,
        colocation_group="decoder-compute",
        require_same_device=True,
    )
    hidden = _contract(
        "routed-hidden-in",
        "hidden-states",
        ("batch", "sequence", hidden_size),
        dtype,
        device,
        revision,
        batch_dimension=0,
        sequence_dimension=1,
    )
    routes = _contract(
        "routed-expert-routes-in",
        "expert-routes",
        ("batch", "sequence", experts_per_token),
        "adapter-exact-route",
        device,
        revision,
        batch_dimension=0,
        sequence_dimension=1,
        route_identity=f"{architecture_id}@{revision}:route",
    )
    output = hidden.model_copy(update={"boundary_id": "routed-hidden-out"})
    return ExecutionComponent(
        component_id="routed-experts",
        component_type=ExecutionComponentType.ROUTED_EXPERTS,
        engine_id=engine_id,
        architecture_id=architecture_id,
        model_revision=revision,
        placement=placement,
        input_contracts=(hidden, routes),
        output_contracts=(output,),
        depends_on=("attention", "router"),
        estimated_memory_bytes=required_memory_bytes,
        metadata=metadata,
    )


__all__ = ["routed_expert_component", "sparse_outer_components"]
