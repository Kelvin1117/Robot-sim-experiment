# DyGRO-VLA → π0.5 One-Shot v2：方法稽核與移植差異

專案：`/common/users/cs2202/dygro_pi05_phase012_20261106/oneshot_paper_v2`。
這份紀錄描述實際程式與提交設計，**不是成功率恢復的證據，也不是完全忠實重現聲明**。
原始 One-Shot SFT 唯讀引用，不重新訓練、不重跑 baseline evaluation。舊正式15% run保留。

## 來源與版本

- 以 `../sources/dygro_pmlr.pdf` 的正式版為主要依據，SHA256 `4c8eb489b09403dccb782c6f1adf27d903332e746da574597b8849cf337d60df`。重新讀取 §4、Appendix A–D、Eq1–14、Algorithm1。
- 同時檢查 https://arxiv.org/html/2605.17486v1 與 https://github.com/EDEM-AI/DyGRO-VLA 。公開repository當前只有展示資產/論文，沒有可直接移植的作者訓練程式或完整hyperparameter表。
- 正式版p6另外要求BC與advantage-filtered update；不能僅憑簡化Algorithm1省略雙buffer、warm-up、target更新及BC/filter。
- 論文引用的BC來源：https://arxiv.org/abs/2106.06860 ；advantage weighting來源：https://arxiv.org/abs/2006.09359 。AWAC實際是advantage-weighted likelihood，**不等於hard positive-Q filter**。本port的BC/filter未公開部分不能聲稱是作者實作或exact AWAC。
- `parameter_provenance.json` 逐一列出完整config的所有leaf values，未公開數值一律標為實驗假設。

## 實際修正

1. 移除舊版`action_slots`與`slot_attention`。它們是隨機新增的action-feature替代路徑，沒有提供原生action hidden states。
2. 新版用π0.5原生action expert讀取action hidden states，與conditioning hidden states、proprioception共同接入Appendix A.1的三分支fusion。action stream以`t=1, action=0`的固定probe讀取；不讀demonstration label，不造成train/live label leakage。此probe是雙stream backbone相容假設。
3. 舊8個latent tokens沒有論文數值依據。新版query數量綁定10-step action horizon，不再另設8-token slot bank；**10也是相容假設，不是論文指定值**。Fusion width256、層數2、頭數4、固定latent noise0.05皆明示實驗數值。
4. 原始權重strict load後，逐一比對所有811個stored tensors與safetensors內容；模型constructor的RoPE/position buffers另行檢查。新增fusion與dimension projection隨機初始化，沒有重新初始化原始SFT參數。
5. 原本Stage I 2000×4次sample只約0.126全資料epoch。新版按真正train split的56,751個有效觀測計算：batch8、29,000 steps=4.088個sample-count epoch equivalents；最大57,000 steps=8.035。這是有界訓練預算，不代表均勻隨機抽樣已完整遍歷每frame，也不是論文未公開的作者schedule。
6. Best Stage I只可從達到minimum training budget的checkpoint選出；不能挑回第1000步的低loss snapshot。Validation執行真LIBERO base policy，關閉residual；不再只靠flow loss和action shape放行。
7. 修正SAC state objective的filter位置：Eq6作用於所有mixed-replay states；不再因offline demo Q較低而刪除對應SAC state。BC才依detached demo advantage篩選。未新增loss種類。
8. Difficulty sampler改為累計empirical binary success rate及論文的asymmetric exponential probability；移除EMA、額外probability floor及強制首輪task coverage。
9. Eq6採用原式unit log-policy coefficient（舊設定0.01沒有論文數值依據）；Eq12不加額外entropy backup。
10. Offline replay改用全部45個training demonstrations/task，與held-out最後5個episodes分離；Stage II 2000完整episodes、每episode40次updates，非舊1000/20預算。它們仍是實驗訓練預算。
11. Stage II validation使用與Stage I gate相同seed8123、initial states20–24、max400 steps；記錄delta_vs_stage1。Training reset25–49，最終evaluation0–19，沒有重用這些states做online reset。
12. Checkpoint retention：latest保存完整續訓狀態；best只保存推論需要的delta/actor。避免每個best複製optimizer及巨大replay，控制NFS個人quota與原子存檔峰值。

## Stage I 對照

| 來源 | 實際路徑 | 做法與界線 |
|---|---|---|
| §4.1 / Eq2–3 | `native.py:OpenPIAdapter.encode/flow_loss`、`run.py:train_stage1` | stochastic encoder、flow-matching action objective加λIB DV；沒有teacher、蒸餾或新KL loss。 |
| A.1 fusion | `dygro/representation.py:FusionLayer` | query self-attention、action/proprio cross-attention、conditioning attention與論文明列的learnable scalar gate。保留attention block的標準Transformer skip；沒有新增繞過IB的base-policy bypass。 |
| Action hidden states | `native.py:OpenPIAdapter.encode` | 原生Gemma action stream hidden states取代random slot approximation；雙stream固定probe與投影是移植假設。 |
| IB stochasticity | `InformationBottleneck.forward` | fixed Gaussian noise。論文未公開encoder distribution/variance；不能宣稱是作者選擇。 |
| DV / Appendix B | `DVCritic.estimate`、`train_stage1` | joint paired samples、同step共享permutation作product marginal、logsumexp；critic上升5次、encoder+action head下降。Appendix B supremum要求ascent；Algorithm1 line9把兩者同寫下降存在記號矛盾，以數學supremum為準並記錄。有限critic/minibatch是MI近似，不是精確MI或證明得到的常數差。 |
| Flow matching | `OpenPIAdapter.flow_loss/base_chunk` | OpenPI native noise/time schedule、target noise−action、Euler去噪；7D環境command loss與valid chunk mask，native action width仍pad32。time/action projections與原生expert原權重繼續訓練。 |
| Offline data | `dataset.py:GoalData` | 官方10個HDF5、每task45train/5validation；obs[i]對actions[i+1:]；partial tail mask、20Hz、episode boundaries、7D action/8D state。 |
| Load / initialization | `native.py:load_base` | strict safetensors load及逐tensor equality、constructor buffer檢查；保存pretrained_load_audit.json。 |
| Joint objective | `run.py:train_stage1` | 同一latent graph參與FM與DV encoder update；DV critic梯度隔離。没有新loss。 |

## Stage II 對照

| 來源 | 實際路徑 | 做法與界線 |
|---|---|---|
| Alg1 line11 | `freeze_for_online/validated_adapter` | Stage I完整凍結；只有通過真環境gate的validated.pt可被Stage II載入，並核對hash/config。 |
| Eq4 | `TaskEmbedding.contrastive_loss` | latent projection及learnable task prototypes、dot-product InfoNCE；task ID只作CL supervision，不輸入router。 |
| Eq5 / A.2 | `ResidualPolicy.forward/sample` | 8個residual mean experts、top2 dense softmax再renormalize、共用std model、tanh與scaleα。採A.2的Gaussian mean-MoE解讀；與Eq5逐expert sample字面寫法不同，論文兩處不完全一致。 |
| Eq6 | `regularization.py:actor_objective` | log-policy−mean ensemble Q，unit entropy coefficient，actor gradient不更新critics。計算的是給定frozen base draw的residual conditional density；不是flow base的不可直接求得marginal action log-density。 |
| Eq7–8 | `DynamicRouter.load_balance`、`actor_objective` | dense minibatch average的negative entropy；RL+λCL CL+λLB LB，正式版另有BC/filter。 |
| Eq9–11 | `ResidualLearner.critic_step`、`CriticEnsemble` | 4個ResMLP/attention-pooling critics、TD+Cal、獨立target ensemble及Polyak。ensemble count與步頻是實驗數值。 |
| Eq12–13 | `bellman_target`、`OnlinePipeline.rollout` | min target critics、γ^actual_duration、discount primitive rewards、true terminal/final observation；不加paper Eq12沒有的soft entropy target。 |
| Eq14 | `calibration_loss`、`GoalData.populate` | offline MC return作floor、max(Q_policy,V_MC)−Q_demo；不是自行換成完整logsumexp CQL。 |
| §4.2 replay/warm-up | `GoalData.populate`、`offline_warmup`、`mixed_batch` | 真offline/online分離、均勻50:50混合、先warm-up critic與actor。 |
| §4.2 sampling | `DifficultySampler` | empirical sr與softmax(max(0,target−sr)/temperature)，沒有額外floor。 |
| BC/filter | `regularization.py:actor_objective` | MSE BC對照所引TD3+BC family；offline demo advantage>0的hard Q-filter是明示未公開細節解讀，不宣稱等於所引AWAC，也不換掉SAC。α、BC權重、advantage threshold、entropy/coefficient等見parameter registry。 |
| Residual scale | `ResidualPolicy.residual_scale` | A.2要求tanh後乘α，但未公布α=0.1；本次0.1保留為實驗假設，不是paper default。殘差與base均在同一normalized action space，再共同unnormalize。Actuator [-1,1] clipping為LIBERO command boundary，clip率會記錄。 |
| Alg1 lines12–40 | `train_stage2`、`OnlinePipeline` | task rollout→base+residual→execute/store→mixed minibatch critic/actor/CL/LB updates；replay保存base draw、augmented action與next base，residual由兩者可重建，router由policy重算並另外記錄log。 |

## 明確未解決／不能宣稱完全忠實的部分

- 論文原backbone是DINOv2/SigLIP/Qwen action-token架構；此任務保留π0.5 One-Shot，因此換成原生PaliGemma/Gemma action stream及必要投影。沒有重建作者backbone。
- Stage I採action expert/projections+fusion更新，PaliGemma保持凍結。**這是記憶體/訓練recipe的實驗選擇，並非論文所證明必須的backbone相容修改，也不是作者已公開的freeze recipe。** Stage II才依論文明確凍結整個Stage I。
- 雙stream固定probe、query數量/latent寬度、固定latent variance、decoder prefix映射、所有數值optimizer/schedule、條件residual entropy、BC/filter詳細公式都必須保留實驗標記。
- 真實機器人Appendix C不在本次LIBERO Goal任務範圍；不宣稱完成sim-to-real重現。
- 600集尚未跑完前，不可宣稱15%已恢復、達到作者分數或優於既有SFT baseline。50集gate只是退化攔截，不是統計充分的性能結論。

## 驗證與Slurm

- CPU tests：33項，覆蓋fusion branch gradients/masks、DV方向、router/entropy Jacobian、critic/actor隔離、chunk return、Cal floor、replay、post-action timing、normalization、checkpoint，以及新增的15%/task-collapse gate拒絕與SAC filter回歸測試。
- GPU smoke必須用正式batch8、相同model/fusion與immutable source snapshot，真HDF5更新、LIBERO timing/normalization、實際actor/critic updates、所有stage及Stage I/II resume通過，才能提交正式鏈。Smoke 2-step數值不代表策略成功率。
- Stage I正式gate：seed8123×10tasks×5initial states20–24=50集，residual=None，max400。整體至少75%、每task至少20%；失敗則最多延長至預先設定的57,000 steps再驗，仍失敗非零退出，不寫validated.json。
- Slurm順序：Stage I → afterok Validation → afterok Stage II → afterok Evaluation。四job先hold，檢查GPU/CPU/RAM/log/依賴後才釋放。無無限auto retry、無baseline job；Stage I續訓是一次有界的既定預算。
- 最終protocol：LIBERO Goal 10tasks×20episodes×3seeds（0/1/2）=600，initial states0–19、400-step limit、env.check_success()，成功立即停止。
- 個人quota不是df共享磁碟餘額。submit至少要求14GiB個人quota headroom；best不保留重複optimizer/replay，latest完整可續訓。本次失敗smoke的暫存checkpoint才可清理；原始SFT、舊正式run與其他研究不動。

提交時另確認叢集禁止明示CPU數量（sbatch實際拒絕並說明），故不設定cpus-per-task；CPU allocation於各job的scontrol紀錄核對。GPU與RAM仍明示。先前被拒絕的提交沒有建立任何job。
