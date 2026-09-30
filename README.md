# Paper Content Extractor

从教师卷和学生卷图片中提取题目结构、答案槽位、教师答案和学生作答。项目只保留当前 VLM 流程，不安装、配置或调用本地 OCR。

教师卷用于建立可复用的逻辑题目树。学生卷继承题目和小问身份，先从学生原图定位和转写；仅在教师页到学生页的单应性配准通过且检测到页面级坐标偏移时，才使用变换后的教师模板校正坐标。

## 当前流程

1. 扫描学科、教师卷、学生卷和物理页顺序。
2. 预处理图片并记录文件指纹。
3. 优先复用通过校验的教师题目树缓存；未命中时使用 VLM 从教师全卷生成章节、题目和小问结构。
4. 每页先用一次整页 VLM 请求定位全部题目和答案区域，再基于固定坐标转写答案。
5. 整页转写达到输出 token 上限时，按题目批次重试转写；定位坐标不重新生成。
6. 校验页码、坐标范围、区域冲突、槽位完整性和结果 Schema。
7. 导出题目树、文档 JSON、区域截图、图示资源和槽位可视化。

运行结果固定包含：

- `recognition_mode: "vlm_only"`
- `ocr_used: false`
- `coordinate_authority: "vlm_original_page_pixels"`，或经单应性变换后的教师模板坐标
- `answer_authority: "vlm_original_page"`

输出仍保留空的 `ocr: []` 字段，便于已有结果消费方平滑升级。项目中没有 OCR 后端或 OCR 降级路径。

![当前 VLM 流程](algorithm_flow.svg)

## 模型配置

当前流程默认使用豆包 Seed 2.1 Turbo 视觉模型，通过火山方舟订阅接口调用：

```bash
export DOUBAO_API_KEY='your-api-key'
export DOUBAO_BASE_URL='https://ark.cn-beijing.volces.com/api/plan/v3'
export DOUBAO_MODEL='doubao-seed-2.1-turbo'
```

正式识别时如果未配置 `DOUBAO_API_KEY`，程序会直接报错。可用 `--model` 为单次运行指定其他模型；这不会修改项目默认模型。

CLI 配置的优先级从高到低为：命令行参数 → 当前进程环境变量 → 私有配置文件 → 代码默认值。`.env.example` 只是配置模板，程序不会自动加载项目目录中的 `.env`。

程序也可从 `~/.config/intelligent-grading-system/doubao.env` 读取密钥。该文件必须仅允许当前用户读取：

```bash
chmod 600 ~/.config/intelligent-grading-system/doubao.env
```

私有配置文件可持久保存以下配置（将密钥替换为自己的值）：

```bash
export DOUBAO_API_KEY='your-api-key'
export DOUBAO_BASE_URL='https://ark.cn-beijing.volces.com/api/plan/v3'
export DOUBAO_MODEL='doubao-seed-2.1-turbo'
```

也可用 `GRADING_SECRET_ENV` 指定其他私有配置文件路径。私有文件仅加载 `DOUBAO_API_KEY`、`DOUBAO_BASE_URL`、`DOUBAO_MODEL` 和 `DOUBAO_RESPONSES_ENDPOINT`；并发、超时等 `EXAM_*` 参数应通过环境变量或命令行设置。若终端已导出旧的 `DOUBAO_MODEL`，请更新它，或执行 `unset DOUBAO_MODEL` 后再运行，以使用私有文件中的值。

## 提示词

模型自然语言指令位于 `exam_pipeline/prompts/`：

- `document_topology.md`：整卷题目拓扑
- `page_topology_recovery.md`：单页构树降级
- `page_text_enrichment.md`：题干文字补全
- `topology_correction.md`：整卷拓扑修正
- `whole_page_geometry.md`：整页坐标定位
- `whole_page_transcription.md`：整页答案转写
- `batch_transcription.md`：达到输出 token 上限后的分批转写

Markdown 文件只保存自然语言指令。题目列表、坐标、失败记录和 JSON Schema 仍由代码在请求时动态拼接。程序通过 `exam_pipeline.prompt_loader.load_prompt()` 以 UTF-8 读取提示词；修改文件后需要重新启动正在运行的 Python 进程。

## 安装

要求 Python 3.8+。

```bash
python3 -m pip install -e .
```

开发环境：

```bash
python3 -m pip install -e '.[dev]'
```

核心依赖只有 NumPy、OpenCV 和 Pillow。

## 数据目录

每个学科目录包含一个 `teacher` 教师卷目录，其余子目录为学生卷。每个文档中的图片按文件名自然排序。

```text
dataset/
└── subject_name/
    ├── teacher/
    │   ├── page_01.jpg
    │   └── page_02.jpg
    ├── student001/
    │   ├── page_01.jpg
    │   └── page_02.jpg
    └── student002/
        ├── page_01.jpg
        └── page_02.jpg
```

学生卷必须有同学科教师卷。页面应完整、方向正确，并使用 `page_01.jpg` 这类补零名称。

## 运行

运行识别：

```bash
python3 homework_extractor.py dataset \
  --subject math_01 \
  --timeout 180 \
  -o reports/math_01_run
```

默认情况下，教师卷全部处理完成后，同学科学生卷最多 3 份并发处理。

教师卷有两个不同阶段：全卷构建逻辑题目树（可命中缓存），以及按页提取题目区域、槽位和答案。`--page-workers` 控制后一个阶段，同时适用于教师卷和学生卷，默认每份卷最多 2 页并发。每页先用一次紧凑的整页纯定位请求统一生成全部坐标，再基于固定坐标做一次整页答案转写；只有整页转写达到输出 token 上限时，才按最多 4 个题目或小问分批转写。转写请求不能生成或修改坐标。`EXAM_VISUAL_BATCH_ITEMS` 仅控制这种 token 上限回退的转写批次大小。

题目树允许父题跨页：如果子题出现在父题原始 `references` 之外的页面，程序会把该子题页面及其短锚点并入父题页面范围。含有构树失败记录的 `DRAFT` 缓存不会复用；缓存版本升级时也会使用新的命名空间重新构树。

单次模型输出上限默认为 32,768 token，包含模型推理和最终 JSON，可通过 `EXAM_MAX_OUTPUT_TOKENS` 永久调整。例如在启动环境中设置 `export EXAM_MAX_OUTPUT_TOKENS=32768`。程序接受 1,024 到 131,072 之间的配置；实际可用上限仍由所选模型和方舟服务决定。

学生页与教师页先使用局部特征和 RANSAC 估计单应性。只有配准质量通过且学生坐标与变换后的模板发生页面级冲突时，模板坐标才可用于校正；配准失败或未经验证的坐标会标记为 `UNCERTAIN`。

默认学生阶段最多同时处理 3 × 2 = 6 个页面任务，各页结果按原页顺序合并。可按接口限流情况调整：

```bash
python3 homework_extractor.py dataset \
  --subject math_01 \
  --student-workers 2 \
  --page-workers 1 \
  -o reports/math_01_run
```

将 `--student-workers 1` 设为 1 可关闭学生卷并发；`--page-workers 1` 可关闭页面并发。并发请求只影响调度，教师题目树、页面校验和最终产物写回顺序不变。

VLM 请求默认使用上传压缩副本，原始图片和输出坐标不变。默认将图片按比例缩放到长边不超过 2300 像素，并以 JPEG 质量 85 上传；副本缓存在 `.exam_pipeline_cache/upload_images/`。可通过 `EXAM_UPLOAD_MAX_LONG_EDGE=0` 关闭缩放压缩，或调整 `EXAM_UPLOAD_JPEG_QUALITY`。

只处理一个学生时，程序仍会先加载对应教师卷：

```bash
python3 homework_extractor.py dataset \
  --subject chinese \
  --student-id student001 \
  -o reports/chinese_student001
```

检查数据分组而不调用模型：

```bash
python3 homework_extractor.py dataset --dry-run -o reports/dry_run
```

如需完全从头识别，清除题目树缓存并使用新的输出目录：

```bash
rm -rf .exam_pipeline_cache
python3 homework_extractor.py dataset -o reports/fresh_run
```

## 超时与视觉重试

`--timeout`（环境变量 `EXAM_REQUEST_TIMEOUT`，默认 180 秒）传给 HTTP 请求的网络操作超时设置，不是整份卷或整次运行的总时限。多次请求和重试会让实际总耗时超过这个值。提高超时只会允许等待更久，不会加快模型推理。

当前重试有不同层次：

- 请求层：对 HTTP 429、5xx 和捕获到的网络异常进行有限重试。
- 定位层：整页纯定位失败或未通过结果校验时，仍以完整页面和完整题目列表有限重试。
- 转写层：先转写整页；只有接口明确返回输出未完成（token 上限）时才拆成题目批次。批次只填充固定区域里的文本、可辨识性和内容类型。

因此，`page_retries` 包含定位/转写失败重试和 token 上限后的分批转写，并不等于接口报错次数。不同学生的页面复杂度、输出长度、模型排队和请求等待时间不同，即使页数相同且同时开始，结束时间也可能相差较大。

学生坐标默认来自学生原页。程序不会把教师坐标直接覆盖到学生卷；它先用局部特征匹配和 RANSAC 计算教师页到学生页的单应性，再比较学生定位与变换后的模板。只有配准质量合格且页面级偏移成立时才应用模板校正。配准失败、比较证据不足或保留学生局部坐标时，相关几何状态为 `UNCERTAIN`。

## 输出

```text
output/
├── manifest.json
├── subject__teacher__teacher.json
├── subject__student__student001.json
├── document_structure/
├── exam_trees/
├── generated_golden/
├── normalized/
├── roi_patches/
├── visual_extraction/
├── diagram_assets/
└── visualizations/
```

`manifest.json` 汇总模型提供方、文档状态、错误、耗时和视觉重试。每份文档 JSON 包含题目树、槽位、答案、坐标证据和质量状态。

`quality.status=NEED_REVIEW` 表示结果已生成，但仍有缺失答案、无效坐标或证据冲突。使用 `--fail-on-review` 可让批处理在存在待复核结果时返回状态码 2。

## 耗时与问题排查

`document_started` 和 `document_completed` 日志表示文档开始和完成；两条日志之间也可能正在等待模型或补充识别。

| 记录位置 | 用途 |
|---|---|
| `manifest.json` 的 `duration_seconds` | 整次运行实际耗时（秒） |
| `manifest.json` 的 `documents[].duration_seconds` | 每份卷实际耗时 |
| `documents[].phase_trace` | 预处理、构树、页级提取等阶段的状态，部分阶段有 `duration_seconds` |
| 每份文档 JSON 的 `performance.events` | 各次操作耗时、成功/失败、异常类型和可用的 token 用量 |
| `visual_extraction/<文档ID>/page_XX.json` 的 `attempts` | 整页定位、整页转写和 token 上限分批转写的处理结果 |

`performance.cumulative_operation_seconds` 是操作耗时累加值。并发调用会重叠，部分操作还可能嵌套，因此它不能作为实际总耗时，也不能直接把各学生卷耗时相加。视觉阶段已经包含答案转写，后续 `answer_recognition` 阶段的整理耗时接近零，不代表模型没有花时间识别答案。

某次调用超时且没有 `usage` 时，仅凭现有记录无法区分服务端排队、推理缓慢或网络延迟。最终 `errors: 0` 表示没有文档最终失败，不代表中间没有请求失败或结果无需复核。

## 本地检查

```bash
python3 -m compileall -q exam_pipeline homework_extractor.py
python3 -m unittest discover -s tests -v
python3 homework_extractor.py --help
python3 homework_extractor.py dataset --subject math_01 --dry-run -o reports/dry_run
```

以上检查不调用模型；`--dry-run` 会生成占位输出，不能验证真实识别质量。自动化测试覆盖整页优先、token 上限分批转写、教师顺序冲突保留、多个空白物理区域合并、单应性配准和不确定坐标映射。
