"""CQL-style loss for DQN in discrete action spaces."""
import ray
from ray.rllib.algorithms.dqn.dqn_tf_policy import PRIO_WEIGHTS
from ray.rllib.algorithms.dqn.dqn_torch_policy import (
    QLoss,
    ComputeTDErrorMixin,
    adam_optimizer,
    before_loss_init,
    build_q_model_and_distribution,
    build_q_stats,
    compute_q_values,
    extra_action_out_fn,
    get_distribution_inputs_and_class,
    grad_process_and_td_error_fn,
    postprocess_nstep_and_prio,
    setup_early_mixins,
)
from ray.rllib.policy.policy import Policy
from ray.rllib.policy.policy_template import build_policy_class
from ray.rllib.policy.sample_batch import SampleBatch
from ray.rllib.policy.torch_mixins import LearningRateSchedule, TargetNetworkMixin
from ray.rllib.utils.annotations import OldAPIStack
from ray.rllib.utils.framework import try_import_torch
from ray.rllib.utils.torch_utils import (
    FLOAT_MIN,
    concat_multi_gpu_td_errors,
    huber_loss,
    l2_loss,
)

torch, nn = try_import_torch()
F = nn.functional if nn else None


@OldAPIStack
def build_cql_losses(policy: Policy, model, _, train_batch: SampleBatch) -> torch.Tensor:
    """Constructs the CQL-augmented loss for DQNTorchPolicy."""
    config = policy.config

    # Q-network evaluation.
    q_t, q_logits_t, q_probs_t, _ = compute_q_values(
        policy,
        model,
        {"obs": train_batch[SampleBatch.CUR_OBS]},
        explore=False,
        is_training=True,
    )

    # Target Q-network evaluation.
    q_tp1, q_logits_tp1, q_probs_tp1, _ = compute_q_values(
        policy,
        policy.target_models[model],
        {"obs": train_batch[SampleBatch.NEXT_OBS]},
        explore=False,
        is_training=True,
    )

    # Q scores for actions which we know were selected in the given state.
    one_hot_selection = F.one_hot(
        train_batch[SampleBatch.ACTIONS].long(), policy.action_space.n
    )
    q_t_clean = torch.where(
        q_t > FLOAT_MIN, q_t, torch.tensor(FLOAT_MIN, device=q_t.device)
    )
    q_t_selected = torch.sum(q_t_clean * one_hot_selection, 1)
    q_logits_t_selected = torch.sum(
        q_logits_t * torch.unsqueeze(one_hot_selection, -1), 1
    )

    # Compute estimate of best possible value starting from state at t + 1.
    if config["double_q"]:
        (
            q_tp1_using_online_net,
            q_logits_tp1_using_online_net,
            q_dist_tp1_using_online_net,
            _,
        ) = compute_q_values(
            policy,
            model,
            {"obs": train_batch[SampleBatch.NEXT_OBS]},
            explore=False,
            is_training=True,
        )
        q_tp1_best_using_online_net = torch.argmax(q_tp1_using_online_net, 1)
        q_tp1_best_one_hot_selection = F.one_hot(
            q_tp1_best_using_online_net, policy.action_space.n
        )
        q_tp1_best = torch.sum(
            torch.where(
                q_tp1 > FLOAT_MIN, q_tp1, torch.tensor(0.0, device=q_tp1.device)
            )
            * q_tp1_best_one_hot_selection,
            1,
        )
        q_probs_tp1_best = torch.sum(
            q_probs_tp1 * torch.unsqueeze(q_tp1_best_one_hot_selection, -1), 1
        )
    else:
        q_tp1_best_one_hot_selection = F.one_hot(
            torch.argmax(q_tp1, 1), policy.action_space.n
        )
        q_tp1_best = torch.sum(
            torch.where(
                q_tp1 > FLOAT_MIN, q_tp1, torch.tensor(0.0, device=q_tp1.device)
            )
            * q_tp1_best_one_hot_selection,
            1,
        )
        q_probs_tp1_best = torch.sum(
            q_probs_tp1 * torch.unsqueeze(q_tp1_best_one_hot_selection, -1), 1
        )

    loss_fn = huber_loss if policy.config["td_error_loss_fn"] == "huber" else l2_loss

    q_loss = QLoss(
        q_t_selected,
        q_logits_t_selected,
        q_tp1_best,
        q_probs_tp1_best,
        train_batch[PRIO_WEIGHTS],
        train_batch[SampleBatch.REWARDS],
        train_batch[SampleBatch.TERMINATEDS].float(),
        config["gamma"],
        config["n_step"],
        config["num_atoms"],
        config["v_min"],
        config["v_max"],
        loss_fn,
    )

    # Conservative Q-Learning penalty for discrete actions.
    temperature = float(config.get("cql_temperature", 1.0))
    temperature = max(temperature, 1e-6)
    min_q_weight = float(config.get("cql_min_q_weight", 1.0))
    cql_logsumexp = torch.logsumexp(q_t_clean / temperature, dim=1) * temperature
    cql_loss = torch.mean(cql_logsumexp - q_t_selected)

    total_loss = q_loss.loss + min_q_weight * cql_loss

    # Store values for stats function in model (tower).
    model.tower_stats["td_error"] = q_loss.td_error
    model.tower_stats["q_loss"] = q_loss
    model.tower_stats["cql_loss"] = cql_loss
    model.tower_stats["cql_total_loss"] = total_loss

    return total_loss


CQLDQNTorchPolicy = build_policy_class(
    name="CQLDQNTorchPolicy",
    framework="torch",
    loss_fn=build_cql_losses,
    get_default_config=lambda: ray.rllib.algorithms.dqn.dqn.DQNConfig(),
    make_model_and_action_dist=build_q_model_and_distribution,
    action_distribution_fn=get_distribution_inputs_and_class,
    stats_fn=build_q_stats,
    postprocess_fn=postprocess_nstep_and_prio,
    optimizer_fn=adam_optimizer,
    extra_grad_process_fn=grad_process_and_td_error_fn,
    extra_learn_fetches_fn=concat_multi_gpu_td_errors,
    extra_action_out_fn=extra_action_out_fn,
    before_init=setup_early_mixins,
    before_loss_init=before_loss_init,
    mixins=[
        TargetNetworkMixin,
        ComputeTDErrorMixin,
        LearningRateSchedule,
    ],
)
