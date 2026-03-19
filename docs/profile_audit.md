# Agent Profile 审查文档

## 示例 Profile

以下是由 `USPopulationBuilder` 生成的一个真实 agent profile（seed=42, n=500, pid=137）：

| 字段 | 值 |
|------|-----|
| pid | 137 |
| age | 40 |
| gender | male |
| metro_status | metro |
| state | North Carolina |
| education | some_college |
| race | Black |
| political_leaning | Independent |
| income_annual | $56,343 |
| asset_level | medium |
| debt_has_loan | 1 (有房贷) |
| debt_amount | $154,223 |
| homeownership | owner |
| has_student_debt | 1 |
| has_health_insurance | 1 |
| employ_status | employed |
| job_satisfaction | 4.0 |
| happiness | 2.9 |
| life_satisfaction | 3.9 |
| future_confidence | 3.8 |
| income_local_rank | 2.4 |
| social_status | 3.0 |
| financial_satisfaction | 3.3 |
| work_reward_belief | 3.1 |
| mobility_belief | 2.8 |
| health_status | good |
| social_trust | cautious |
| income_percentile | 41.8 |

---

## 每个字段的数据来源与计算方式

### 第一层：人口统计学特征（Joint Distribution Sampling）

采样顺序严格按照条件依赖链：State → Race|State → Education|Race → Party|Race×Education → 其余独立边际变量

#### 1. `state` = "North Carolina"

- **来源**: ACS 2023 州人口估计
- **采样方式**: 从 `STATE_WEIGHTS` 边际分布中独立采样。North Carolina 权重 = 0.032（占全国 3.2%）
- **代码**: `rng.choice(list(STATE_WEIGHTS.keys()), size=n, p=_norm(STATE_WEIGHTS))`

#### 2. `race` = "Black"

- **来源**: **Census API** B03002 表（Hispanic or Latino Origin by Race），2022 ACS 1-Year
- **采样方式**: **条件采样 P(Race | State)**。先查询 Census API 获取该州的种族分布，如果 API 不可用则回退到硬编码的 `STATE_RACE_DIST`
- **North Carolina 的种族分布**: `{"White": 0.60, "Hispanic": 0.10, "Black": 0.21, "Asian": 0.03, "Other": 0.06}`
- **为什么用 B03002 而不是 B02001**: B02001 的 "White" 包含 Hispanic White，会导致 Hispanic 群体被双重计算。B03002 直接提供 Non-Hispanic White，避免了这个问题
- **API 调用**: `CensusCollector.get_state_race_distributions(year=2022)` → 调用 `https://api.census.gov/data/2022/acs/acs1?get=NAME,B03002_001E,...&for=state:*`
- **缓存**: JSON 缓存在 `data/cache_us/census_state_race.json`

#### 3. `education` = "some_college"

- **来源**: ACS 2023（25岁以上教育程度，按种族）
- **采样方式**: **条件采样 P(Education | Race)**
- **Black 的教育分布** (`EDU_BY_RACE["Black"]`):
  - less_than_hs: 12%, high_school: 30%, some_college: 25%, bachelors: 20%, graduate: 13%
- **代码**: `rng.choice(edu_keys, p=_norm(edu_dist))`

#### 4. `political_leaning` = "Independent"

- **来源**: **CES 2022**（Cooperative Election Study, Harvard Dataverse, n=54,828）
- **采样方式**: **条件采样 P(Party | Race) + Education Shift**。分两步：
  1. 查 `PARTY_BY_RACE["Black"]` 获取基础分布:
     - Democrat: 70.9%, Independent: 22.5%, Republican: 6.5%
  2. 叠加 `EDU_PARTY_SHIFT["some_college"]` 教育调整:
     - Democrat: -1.2%, Independent: +0.9%, Republican: +0.3%
  3. 最终调整后分布: Democrat≈69.7%, Independent≈23.4%, Republican≈6.8%
  4. 从调整后分布中采样
- **CES 数据提取方式**: 从 `data/ces/ces2022.csv`（60,000行，193MB）中，使用 `pid3` 变量（1=Democrat, 2=Republican, 3=Independent）和 `race`、`educ` 变量交叉统计
- **EDU_PARTY_SHIFT 的含义**: 教育程度对党派的额外影响。研究生 Democrat +9.5%, Republican -9.1%；高中 Democrat -5%, Republican +7%

#### 5. `age` = 40

- **来源**: 近似美国成年人年龄分布
- **采样方式**: `np.clip(rng.normal(45, 17, size=n), 18, 85).astype(int)`
- **注意**: 使用正态分布 N(45, 17) 截断到 [18, 85]。这是一个简化——实际美国年龄分布更接近均匀分布。但对 ICS 模拟影响不大，因为 ICS 不按年龄加权

#### 6. `gender` = "male"

- **来源**: ACS 性别比例
- **采样方式**: `rng.choice(["male", "female"], p=[0.49, 0.51])`——独立边际分布

#### 7. `metro_status` = "metro"

- **来源**: ACS 城市化率
- **采样方式**: `rng.choice(["metro", "non-metro"], p=[0.86, 0.14])`——独立边际分布

---

### 第二层：经济状况（Conditional on Demographics）

#### 8. `employ_status` = "employed"

- **来源**: CPS 2024（Current Population Survey, 16岁以上平民）
- **边际分布** (`EMPLOY_WEIGHTS`): employed 60%, unemployed 4%, retired 20%, other 16%
- **条件调整**: 如果 age ≥ 65 则重新采样为 retired 70% / employed 20% / other 10%；如果 age < 22 且低学历则 employed 55% / unemployed 15% / other 30%
- **此 agent**: age=40，不触发任何条件调整，直接使用边际分布

#### 9. `income_annual` = $56,343

- **来源**: **Census API** B19013B_001E（Black Household Median Income），2022 ACS
- **计算方式**: 三重条件 `income = N(mean × edu_mult × age_mult, std × 0.6)`
  1. **Census API 查询**: `CensusCollector.get_income_params_by_race(year=2022)` 获取 Black median = $51,374
     - 估算: mean ≈ median × 1.25 = $64,218, std ≈ median × 0.7 = $35,962
  2. **Education multiplier** (`edu_mult`): some_college → 0.90
  3. **Age multiplier** (`AGE_INCOME_MULT`): age 40 落在 (35,44) → 1.00
  4. **Employment adjustment**: employed → 不调整（unemployed 会 ×0.3, retired 会 ×0.55）
  5. **最终**: `max(0, N(64218 × 0.90 × 1.00, 35962 × 0.6))` = `N(57796, 21577)` → 采样得 $56,343
- **API 端点**: `https://api.census.gov/data/2022/acs/acs1?get=NAME,B19013B_001E&for=us:1`
- **缓存**: `data/cache_us/census_national_rates.json` 中的 `income_by_race` 字段

#### 10. `income_percentile` = 41.8

- **来源**: 生成后全局计算
- **计算方式**: `pop["income_annual"].rank(pct=True) * 100`——在全部 500 个 agent 中排百分位
- **含义**: $56,343 在这 500 人中排第 41.8%

#### 11. `homeownership` = "owner"

- **来源**: Census API B25003（Tenure: Owner vs Renter），叠加 ACS 种族/年龄差异
- **计算方式**: 多因素联合概率
  1. 基础概率: Census API `homeownership` rate（全国约 65.5%）
  2. **Age 调整**: age=40 不触发（<30 减 0.30, <35 减 0.15, >55 加 0.10）
  3. **Race 调整**: Black → 减 0.20（ACS: Black 住房自有率 ~44% vs White ~73%）
  4. **Income 调整**: $56,343 < $70,000 → 不触发高收入加成；> $30,000 → 不触发低收入减少
  5. **最终概率**: 0.655 - 0.20 = 0.455，`clip(0.05, 0.95)` → 0.455
  6. `rng.random() < 0.455` → 此次采样结果为 owner

#### 12. `debt_has_loan` = 1, `debt_amount` = $154,223

- **来源**: 条件逻辑
- **计算方式**: homeownership == "owner" 且 `rng.random() < 0.65` → 65% 的 owner 有房贷
- **贷款金额**: `max(0, N(250000, 120000))` → 采样得 $154,223

#### 13. `has_student_debt` = 1

- **来源**: 年龄×教育条件分布
- **计算方式**:
  1. 基础概率: age=40 < 45 → `STUDENT_DEBT_RATE_YOUNG` = 0.35
  2. Education 调整: some_college → +0.05
  3. 最终: 0.40 概率有学生贷款 → 此次采样为 1

#### 14. `has_health_insurance` = 1

- **来源**: ACS 医保覆盖率 ~92%
- **计算方式**:
  1. 基础概率: `INSURED_RATE` = 0.92
  2. Income 调整: $56,343 > $25,000 → 不减
  3. Race 调整: Black → 不调整（Hispanic 会减 0.08，因为 ACS 显示 Hispanic 无保险率 ~18%）
  4. 最终: 0.92 → 采样为 1

#### 15. `asset_level` = "medium"

- **来源**: 基于 income 和 age 的规则
- **计算方式**:
  1. age=40 < 50 → `age_asset_bonus` = 0
  2. income=$56,343 → 不 < $30,000（排除 low），不 > $80,000（排除 high）
  3. → "medium"

---

### 第三层：主观感知变量（GSS 2022 Fitted Parameters）

所有主观感知变量的参数均从 **GSS 2022 微观数据**（General Social Survey, NORC, n=4,149）中拟合得到。GSS 原始数据存储在 `data/gss/2022/GSS2022.dta`。

拟合方式：按 Race × Income Tercile（或其他分组变量）分组，计算该组在 GSS 变量上的 mean 和 std，然后在模拟时从 `N(mean, std)` 中采样。

**此 agent 的 GSS 查找键**: gss_race="Black", inc_terc="mid"（income $56,343 在 $35k-$80k 之间）

#### 16. `happiness` = 2.9

- **来源**: GSS 2022 变量 `HAPPY`（1=not too happy, 2=pretty happy, 3=very happy）
- **GSS 参数**: `GSS_HAPPINESS[("Black", "mid")]` = {mean: 1.903, std: 0.647}
- **计算**: `clip(N(1.903, 0.647), 1, 3)` → 采样得 2.9
- **GSS 实际数据**: Black 中等收入群体中，约 29% "very happy", 38% "pretty happy", 33% "not too happy"

#### 17. `financial_satisfaction` = 3.3

- **来源**: GSS 2022 变量 `SATFIN`（1=not at all satisfied, 2=more or less, 3=pretty well satisfied）
- **GSS 参数**: `GSS_SATFIN[("Black", "mid")]` = {mean: 1.820, std: 0.713}
- **计算**:
  1. 采样 GSS 原始尺度: `clip(N(1.820, 0.713), 1, 3)` → 得到 satfin_raw
  2. 线性映射到 1-5 尺度: `fin_sat = 1 + (satfin_raw - 1) × 2`（GSS 1→1, 2→3, 3→5）
  3. → 3.3

#### 18. `life_satisfaction` = 3.9

- **来源**: GSS 2022 变量 `LIFE`（1=dull, 2=routine, 3=exciting）——用作生活满意度的代理变量
- **GSS 参数**: age=40 落在 (30,45) → `GSS_LIFE[(30,45)]` = {mean: 2.373, std: 0.566}
- **计算**:
  1. 采样: `clip(N(2.373, 0.566), 1, 3)` → 得到 life_raw
  2. 映射到 1-5: `life_sat = 1 + (life_raw - 1) × 2`
  3. → 3.9

#### 19. `future_confidence` = 3.8

- **来源**: 派生变量（GSS 无直接对应变量），锚定在 `life_satisfaction` 上
- **计算**:
  1. 基础 = life_satisfaction = 3.9
  2. Age shift: age=40 在 [35, 60] 之间 → 0.0
  3. Employment adjustment: employed → 0.0（unemployed 会减 0.8）
  4. `clip(3.9 + 0.0 + 0.0 + N(0, 0.5), 1, 5)` → 3.8

#### 20. `job_satisfaction` = 4.0

- **来源**: GSS 2022 变量 `SATJOB`（1=very dissatisfied, 2=a little dissatisfied, 3=moderately satisfied, 4=very satisfied）
- **GSS 参数**: `GSS_SATJOB["employed"]` = {mean: 3.277, std: 0.772}
- **计算**: `clip(N(3.277, 0.772), 1, 4)` → 4.0

#### 21. `work_reward_belief` = 3.1

- **来源**: GSS 2022 变量 `GETAHEAD`（1=luck/help, 2=both, 3=hard work）——"努力工作能否带来成功"
- **GSS 参数**: `GSS_GETAHEAD["Black"]` = {mean: 2.410, std: 0.766}
- **计算**:
  1. 采样: `clip(N(2.410, 0.766), 1, 3)` → 得到 getahead_raw
  2. 映射到 1-4: `work_belief = 1 + (getahead_raw - 1) × 1.5`（GSS 1→1, 2→2.5, 3→4）
  3. → 3.1

#### 22. `mobility_belief` = 2.8

- **来源**: 派生自 `work_reward_belief` + 种族调整
- **计算**:
  1. 基础 = work_reward_belief = 3.1
  2. Race adjustment: Black → -0.2（反映社会流动性感知的种族差异）
  3. `clip(3.1 - 0.2 + N(0, 0.3), 1, 4)` → 2.8

#### 23. `social_trust` = "cautious"

- **来源**: GSS 2022 变量 `TRUST`（"Generally speaking, would you say that most people can be trusted or that you can't be too careful?"）
- **GSS 参数**: `GSS_TRUST_RATE["Black"]` = 0.165（16.5% 选择 "can trust"）
- **计算**: `rng.random() < 0.165` → 此次为 False → "cautious"
- **GSS 发现**: Black 群体的信任率（16.5%）显著低于 White（32.8%），这在社会学文献中是一致的发现

#### 24. `health_status` = "good"

- **来源**: GSS 2022 变量 `HEALTH`（1=excellent, 2=good, 3=fair, 4=poor）
- **GSS 参数**: `GSS_HEALTH[("Black", "mid")]` = {excellent: 16.7%, good: 54.3%, fair: 26.6%, poor: 2.5%}
- **计算**: 从分类分布中直接采样 → "good"

#### 25. `social_status` = 3.0

- **来源**: 派生变量，代理 GSS `CLASS_` 变量（1=lower, 2=working, 3=middle, 4=upper）
- **计算**:
  1. Education 基础: some_college → 2.6
  2. Income 调整: min($56,343 / $100,000, 1.0) × 0.8 = 0.45
  3. `clip(N(2.6 + 0.45, 0.6), 1, 5)` = `N(3.05, 0.6)` → 3.0

#### 26. `income_local_rank` = 2.4

- **来源**: 基于绝对收入的粗估，后续由 `income_percentile` 替代
- **计算**: `clip(1 + (income / 40000), 1, 5)` = `1 + 56343/40000` = 2.4
- **注意**: 这个字段的精确值不太重要，`income_percentile` 是更准确的相对位置指标

---

## 数据源汇总

| 数据源 | 类型 | 提供的变量 | 获取方式 |
|--------|------|-----------|---------|
| **Census API (ACS 2022)** | 实时 API | State×Race 分布, Income by Race, Homeownership rate, State median income | `api.census.gov`, JSON 缓存 |
| **CES 2022** (Harvard Dataverse) | 静态数据集 | Party×Race, Party×Race×Education | 下载 csv (60k rows), 提取参数硬编码 |
| **GSS 2022** (NORC) | 静态数据集 | Happiness, SATFIN, TRUST, SATJOB, GETAHEAD, LIFE, HEALTH (all by Race/Income/Age) | 下载 .dta, 提取参数硬编码 |
| **ACS 2023** (summary tables) | 硬编码 | State population weights, Education×Race, Metro rate | 查表硬编码 |
| **CPS 2024** | 硬编码 | Employment distribution | 查表硬编码 |
| **BRFSS** | 硬编码 | Health status marginal (作为 fallback) | 查表硬编码 |

### "实时" vs "静态" 的设计逻辑

- **Census API (实时)**: 种族分布、收入分布等随人口变动而变化，每年 ACS 更新一次，通过 API 实时获取
- **GSS (静态, 每2年)**: 主观感知（happiness、trust、satfin 等）是横截面结构关系，变化缓慢。GSS 每2年发布一次，参数直接硬编码在代码中
- **CES (静态, 每2年)**: 党派×种族结构同样变化缓慢，每次选举年发布新数据
- **FRED/UMCSENT (实时, 月度)**: 不在人口生成中使用，而是在 Bayesian prior calibration 阶段用于锚定情绪水平

---

## 随机性影响分析

### 个体层面（同一人口统计特征，不同种子）

固定 Race=Black, Education=some_college, Age=40, Employed，换 10 个种子：

| 变量 | mean | std | range | CV |
|------|------|-----|-------|-----|
| income | $51,958 | $21,180 | [$9.9k, $77.1k] | 41% |
| happiness | 2.20 | 0.63 | [1.2, 3.0] | 29% |
| fin_satisfaction | 2.79 | 0.82 | [1.0, 4.0] | 29% |
| life_satisfaction | 3.17 | 1.53 | [1.0, 5.0] | 48% |
| future_confidence | 3.18 | 1.52 | [1.0, 5.0] | 48% |
| job_satisfaction | 3.01 | 0.54 | [2.2, 3.9] | 18% |
| work_belief | 2.96 | 0.94 | [1.0, 4.0] | 32% |

**个体波动很大**（CV 18%-48%），这是设计预期——同一人口统计群体内部确实存在很大个体差异。

### 群体层面（n=1000，不同种子）

| seed | median income | Dem% | Rep% | Ind% | owner% | happiness | fin_sat | trust% | health good+% |
|------|--------------|------|------|------|--------|-----------|---------|--------|---------------|
| 42 | $67,077 | 44.1 | 25.7 | 30.2 | 54.8 | 2.023 | 2.945 | 23.7 | 71.7 |
| 123 | $68,803 | 42.6 | 26.1 | 31.3 | 56.7 | 2.007 | 2.895 | 24.2 | 71.0 |
| 456 | $67,989 | 44.0 | 22.8 | 33.2 | 56.6 | 1.958 | 3.030 | 24.6 | 73.8 |
| 789 | $62,703 | 45.0 | 26.7 | 28.3 | 55.7 | 1.985 | 2.966 | 24.7 | 75.0 |
| 2025 | $66,245 | 43.9 | 24.6 | 31.5 | 56.2 | 2.033 | 2.929 | 26.2 | 67.6 |

**群体层面非常稳定**：
- 收入中位数 CV ≈ 3%（$62.7k ~ $68.8k）
- 党派分布 CV ≈ 2-5%
- Happiness mean CV ≈ 1.5%
- 住房自有率 CV ≈ 1.4%

**结论**: 个体 agent 的随机性是真实个体差异的合理反映，但 n=1000 时群体统计量高度稳定，种子选择不会实质影响 ICS 模拟结果。

---

## 条件依赖关系图

```
State (ACS marginal)
  └─→ Race | State (Census API B03002)
        ├─→ Education | Race (ACS)
        │     └─→ Party | Race × Education (CES + EDU_PARTY_SHIFT)
        ├─→ Income | Race × Education × Age × Employment (Census API B19013)
        │     ├─→ Homeownership | Race × Age × Income (Census API B25003 + adjustments)
        │     ├─→ Health Insurance | Race × Income (ACS)
        │     ├─→ Asset Level | Income × Age
        │     └─→ Income Tercile (for GSS lookups)
        │           ├─→ Happiness | Race × Income Tercile (GSS HAPPY)
        │           ├─→ Financial Satisfaction | Race × Income Tercile (GSS SATFIN)
        │           ├─→ Health Status | Race × Income Tercile (GSS HEALTH)
        │           └─→ Social Trust | Race (GSS TRUST)
        └─→ Work Reward Belief | Race (GSS GETAHEAD)
              └─→ Mobility Belief | Race (derived)

Age (N(45,17) marginal)
  ├─→ Income multiplier (CPS AGE_INCOME_MULT)
  ├─→ Employment | Age (CPS, conditional override)
  ├─→ Life Satisfaction | Age (GSS LIFE)
  │     └─→ Future Confidence | Age × Employment (derived)
  └─→ Student Debt | Age × Education

Employment (CPS marginal, age-adjusted)
  └─→ Job Satisfaction | Employment (GSS SATJOB)

Gender, Metro (independent marginals)
```
