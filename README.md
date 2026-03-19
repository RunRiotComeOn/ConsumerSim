# ConsumerSim

面向美国消费者信心的仿真项目。当前代码库的实际可执行版本是 `main_us.py`，目标是生成 Michigan 风格的消费者信心指标：

- `ICS` (`Index of Consumer Sentiment`)
- `ICC` (`Index of Current Conditions`)
- `ICE` (`Index of Consumer Expectations`)

项目通过合成人口样本、抽取一部分核心代理做 LLM 推断，再用贝叶斯后验把结果扩展到全部代理，最终输出指数、验证结果和图表。

## 当前可执行入口

主入口:

```bash
python main_us.py
```

辅助诊断脚本:

```bash
python diagnose_llm.py --month 2025-03 --agents 100
```

## 项目结构

```text
ConsumerSim/
├─ config/
│  └─ config_us.yaml               # 当前使用的配置
├─ data/
│  ├─ cache_us/                    # 宏观数据/人口缓存
│  ├─ ces/                         # CES 参数与原始数据
│  └─ gss/                         # GSS 参数与原始数据
├─ deps/                           # 本地 vendor 依赖目录
├─ docs/
│  └─ profile_audit.md
├─ notebooks/                      # 研究/实验 notebook
├─ results_us/
│  ├─ monthly_backtest.csv         # 月度回测结果
│  └─ figures/                     # 图表输出目录
├─ src/
│  ├─ agents/
│  │  ├─ base.py                   # 通用代理与响应结构
│  │  └─ base_us.py                # USConsumerAgent
│  ├─ behavior_model/
│  │  ├─ demo_guidance_us.py       # 人口学先验/映射规则
│  │  └─ predictor_us.py           # 贝叶斯后验扩展
│  ├─ data/
│  │  ├─ macro_context.py          # 宏观上下文组装
│  │  └─ *_collector.py            # FRED / Census / 新闻等采集器
│  ├─ evaluation/
│  │  ├─ micro_us.py               # 微观验证
│  │  └─ structural_us.py          # 结构验证
│  ├─ llm/
│  │  ├─ inference_us.py           # OpenAI 兼容推理封装
│  │  ├─ prompts_us.py             # 提示词构造
│  │  └─ response_parser.py        # LLM 响应解析
│  ├─ population/
│  │  └─ builder_us.py             # 美国人口合成
│  ├─ simulation/
│  │  ├─ engine_us.py              # 单次仿真引擎
│  │  ├─ monthly_runner.py         # 月度/回测/预测
│  │  └─ index.py                  # ICS/ICC/ICE 计算
│  └─ visualization/
│     ├─ plots.py                  # 通用分布/分组图
│     └─ plots_us.py               # US 专用图表
├─ diagnose_llm.py                 # 诊断指定月份的 LLM 输出
├─ main_us.py                      # 主入口
├─ README.md
└─ requirements.txt                # Python 依赖
```

## 运行流程

当前代码里的实际流程如下:

1. `USPopulationBuilder` 根据 ACS / CPS / GSS / CES 等数据构建美国合成人口。
2. `USSimulationEngine.build_agents()` 创建代理，并按 `core_agent_ratio` 抽取核心代理。
3. 如果不是 `--no-llm`，会先用最新或指定目标的 Michigan `UMCSENT` 对先验做校准。
4. 单次运行会拉取宏观上下文；月度运行会按目标月份构建该月可见的数据窗口。
5. `USLLMInferenceEngine` 只对核心代理做 LLM 推断。
6. `USBayesianPredictor` 用核心代理回答更新后验，再为普通代理补全回答。
7. `compute_indices()` 计算 `ICS / ICC / ICE` 以及各题目的相对指数。
8. 单次运行会导出 CSV、验证结果和图表；月度回测会输出 `monthly_backtest.csv`。

## 支持的运行模式

### 1. 单次仿真

```bash
python main_us.py
python main_us.py --agents 3000
python main_us.py --no-llm
python main_us.py --config config/config_us.yaml
```

说明:

- 默认 `--mode single`
- `--no-llm` 会跳过 LLM 调用，改用 `_inject_synthetic_responses()` 生成测试回答
- 单次模式会输出结果表、验证表和图表

### 2. 指定月份仿真

```bash
python main_us.py --mode monthly --month 2025-06
python main_us.py --mode monthly --month 2025-06 --agents 1000
```

说明:

- 实际执行逻辑在 `src/simulation/monthly_runner.py`
- 会按目标月份构造 survey window，避免直接看未来数据
- 会优先用前一个月的真实 `UMCSENT` 作为先验锚点

### 3. 历史回测

```bash
python main_us.py --mode backtest --start 2024-06 --end 2026-01
```

说明:

- 按月循环执行
- 输出 `results_us/monthly_backtest.csv`
- 若有足够真实值，会打印相关系数、`MAE`、`RMSE`

### 4. 当月预测

```bash
python main_us.py --mode forecast
```

说明:

- 当前实现调用 `MonthlyRunner.forecast_next_month()`
- 实际上是对“当前月份”执行一次月度仿真

## 诊断脚本

`diagnose_llm.py` 用来排查某个月的 LLM 推断是否合理:

```bash
python diagnose_llm.py --month 2025-03 --agents 100
```

它会输出:

- 发送给 LLM 的宏观上下文
- 核心代理回答分布
- 核心代理 `ICS`
- 后验扩展后的最终 `ICS`
- 目标月份真实 `ICS`（如果能取到）
- 部分核心代理画像样例

## 配置说明

当前运行配置文件是 `config/config_us.yaml`，主要包含以下部分:

### `data`

- `processed_dir`: 处理后数据目录
- `cache_dir`: 缓存目录
- `fred_api_key`: FRED key
- `census_api_key`: Census API key
- `news_api_key`: NewsAPI key
- `nyt_api_key`: New York Times Archive API key
- `guardian_api_key`: Guardian API key
- `bing_api_key`: Bing News Search key；代码支持，但当前配置文件默认未填写
- `bing_endpoint`: Bing 接口地址
- `news_provider`: `nyt_archive | nyt | guardian | bing | newsapi`
- `nyt_us_only`, `guardian_us_only`: 新闻过滤选项
- `use_ustr_signal`, `use_gdelt_signal`, `use_news_signal`, `use_trends_signal`: 是否启用对应宏观信号
- `trends_geo`, `trends_max_batches`: Google Trends 配置
- `authoritative_news_domains`: NewsAPI 允许的新闻域名白名单

### `simulation`

- `total_agents`: 总代理数
- `core_agent_ratio`: 核心代理比例
- `batch_size`: 每批 LLM 推理代理数
- `random_seed`: 随机种子
- `survey_window_cutoff_day`: 调查窗口截断日
- `data_integrity_fail_on_violation`: 月度模式下数据完整性不满足时是否直接失败

### `llm`

- `provider`: 当前默认是 `openai`
- `model`: 当前默认是 `gpt-4o-mini`
- `base_url`: OpenAI 兼容接口地址；留空时走默认 OpenAI 地址
- `api_key`: 支持直接写在配置中的 key
- `api_key_env`: 也支持从环境变量读取，当前字段是 `OPENAI_API_KEY`
- `max_retries`, `temperature`, `timeout`, `max_tokens`

### `posterior`

- `prior_strength`: 后验先验强度

### `index`

- `icc_components`
- `ice_components`
- `ics_components`

### `output`

- `results_dir`
- `figures_dir`

## 输出文件

当前代码默认会写到 `results_us/`:

- `results_us/us_simulation_results.csv`
- `results_us/us_micro_validation.csv`
- `results_us/us_structural_validation.csv`
- `results_us/monthly_backtest.csv`
- `results_us/figures/us_agent_profiles.png`
- `results_us/figures/us_response_distribution.png`
- `results_us/figures/us_demographic_breakdown.png`
- `results_us/figures/us_validation_summary.png`
- `results_us/figures/*.png`

## 数据与缓存

当前项目会使用或生成这些本地数据:

- `data/cache_us/*.parquet`
- `data/cache_us/*.json`
- `data/cache_us/*.xlsx`
- `data/ces/ces2022.csv`
- `data/ces/ces_party_params.json`
- `data/gss/2022/GSS2022.dta`
- `data/gss/gss_params.json`

其中:

- 人口缓存用于加速合成人口生成
- 各类 `collector` 会把宏观数据和新闻摘要缓存在 `data/cache_us`
- `deps/` 是当前仓库内的本地依赖目录，不属于核心源码结构，但运行时会被导入使用

## 安装依赖

```bash
python -m pip install -r requirements.txt
```

当前 `requirements.txt` 中的主要依赖包括:

- `pandas`
- `numpy`
- `pyreadstat`
- `openpyxl`
- `pyyaml`
- `requests`
- `pytrends`
- `scikit-learn`
- `xgboost`
- `openai`
- `scipy`
- `pydantic`
- `matplotlib`

## 运行前建议

- 先检查 `config/config_us.yaml` 中的路径是否适合当前机器。
- 如果不想在配置文件里放密钥，优先改成环境变量方式。
- 首次跑建议先用较小样本，例如 `--agents 100` 或 `--agents 500`。
- 如果只想验证流程是否打通，可以先跑 `--no-llm`。

## 当前 README 与代码对齐说明

这份 README 按当前代码实际行为整理，重点对齐了下面几点:

- 项目当前只有 US 主入口，没有其他国家入口
- 主执行脚本是 `main_us.py`
- 月度、回测、预测都由 `MonthlyRunner` 驱动
- 诊断入口 `diagnose_llm.py` 已纳入说明
- 配置项名称按 `config/config_us.yaml` 当前字段更新
- `llm` 默认值已按当前配置更新为 `openai / gpt-4o-mini`
- 输出文件名按当前代码实际导出路径更新
