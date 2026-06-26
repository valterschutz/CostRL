# Paper Config Provenance

These configs target the belief-based hierarchical PPO experiments from Li and Oliva (2025), "Towards Cost Sensitive Decision Making". The repository requires several operational fields that are not reported in the paper; those are documented separately below.

## Values Taken From The Paper

Source: Appendix D.1, benchmark environments.

- `environment.name`: maps the paper's modified CartPole-v1 and Sepsis simulator to this repo's registered `cartpole-v0` and `sepsis-v0` environments.
- CartPole costs: `0.005`, `0.01`, `0.015`.
- Sepsis costs: `0.005`, `0.01`, `0.02`.
- `environment.max_episode_length`: `500` for CartPole-v1, from the Gym environment cap used by this wrapper; `30` for Sepsis, from the paper's maximum treatment steps.

Source: Appendix D.1 and Appendix D experiment settings.

- `agent.name`: `batch_hier_mbppo` and `seque_hier_mbppo` are the repo implementations corresponding to belief-based batch and sequential CS-HPPO with POSS.
- `agent.max_history_length`: `8`, from the previous 8 observations used for history-based inputs and POSS histories.

Source: Table D.1, PPO hyperparameters.

- `agent.gamma`: `0.99`.
- `agent.gae_lambda`: `0.95`.
- `agent.ratio_clip`: `0.2`.
- CartPole reward weights: `omega_e=1.0`, `omega_v=0.01`, `omega_a=100.0`.
- Sepsis reward weights: `omega_e=1.0`, `omega_v=1.0`, `omega_a=1.0`.
- In these YAML files, those weights are mapped to `terminal_reward_weight`, `tsk_value_term_weight`, and `mod_impute_term_weight` because that is how this repo names the corresponding CS-HPPO terminal/intrinsic reward terms.

Source: Table D.1 and Appendix D.2, POSS implementation.

- `model.latent_dim`: `64`.
- `model.categorical_embed_dim`: `16` for Sepsis categorical features.
- `model.action_embed_dim`: `16`.
- `model.time_embed_dim`: `16`.
- `model.peq_encode_dims`: `[128, 128, 128, 128]`, representing 4 Set Transformer layers with 128 hidden units.
- `model.peq_decode_dims`: `[128, 128, 128, 128]`, representing the decoder Set Transformer stack.
- `model.prior_trans_dims`: `[128, 128, 128, 128]`, representing 4 rational-quadratic coupling transformations with 128 hidden units.
- `model.posterior_trans_dims`: `[128, 128, 128, 128]`, representing 4 rational-quadratic autoregressive transformations with 128 hidden units.

Source: Table D.1 and Appendix D.3, policies.

- `policy.actor_layers`: `[64, 64]`.
- `policy.critic_layers`: `[64, 64]`.
- `policy.belief_embed_type`: `ensemble`, from the paper's statement that set/belief policy outputs are ensembled across set elements.
- `policy.categorical_embed_dim`: `16` for Sepsis categorical features.

Source: Table D.1, training hyperparameters.

- `running.lr_model`: `0.0001`.
- `running.lr_afa`: `0.0003`.
- `running.lr_tsk`: `0.0003`.
- `running.grad_norm`: `1.0`.

## Values Inferred From This Repository

These are not written verbatim in the paper, but they are necessary to select the matching code paths in this repo.

- `model.name`: `poex_con_dis` for continuous CartPole observations and `poex_cat_dis` for categorical Sepsis observations.
- `agent.terminal_reward_type`: `hybrid`, because the paper combines value/entropy/imputation-style reward terms and this repo exposes that combination through `hybrid`.
- `agent.tsk_entropy_term_weight`: `1.0`, because the repo requires this field for `hybrid`; the paper reports `omega_e=1.0` in both environments.

## Values Not Specified In The Paper

These are runnable defaults chosen to satisfy this repo's config schema. They should be adjusted if exact training budgets or implementation details become available.

- `agent.vf_weight`: `0.5`.
- `agent.ent_weight`: `0.01`.
- `agent.num_belief_samples`: `10`.
- `model.peq_embed_dim`: `0`, meaning reuse the prior encoder's permutation-equivariant embedding in this repo.
- `model.shared_prior_posterior`: `false`.
- `model.num_heads`: `4`.
- `model.num_inds`: `16`.
- `model.use_ln`: `false`.
- `model.kl_weight`: `1.0`.
- `model.num_posterior_samp_train`: `1`.
- `model.num_posterior_samp_test`: `10`.
- `policy.belief_embed_dims`: `[64]`.
- `running.exp_dir`: experiment output path.
- `running.device`: `cpu`.
- `running.seed`: `0`; the paper reports three random seeds but does not list the seed values.
- `running.batch_size`: `256`.
- `running.train_env_num`: `16`.
- `running.steps_per_collect`: `10`.
- `running.stage1_iterations`: `1000`.
- `running.stage2_iterations`: `1000`.
- `running.stage3_iterations`: `1000`.
- `running.freeze_model`: `false`, only needed by `seque_hier_mbppo` in this repo.
- `running.validation_freq`: `10`.
- `running.num_valid_episodes`: `100`.
- `running.num_test_episodes`: `100`.
