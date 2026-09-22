# Paper Content Extractor

从教师卷和学生卷图片中提取题目结构、答案槽位、教师答案和学生作答。项目只保留当前 VLM 流程，不安装、配置或调用本地 OCR。

教师卷用于建立可复用的逻辑题目树。学生卷继承题目和小问身份，但题目区域、答案区域、坐标和答案文本均由学生原图独立识别。

## 当前流程

1. 扫描学科、教师卷、学生卷和物理页顺序。
2. 预处理图片并记录文件指纹。
3. 使用 VLM 从教师全卷生成章节、题目和小问结构。
4. 使用 VLM 在每份试卷原页中联合提取题目区域、槽位、坐标和答案。
5. 对缺失页或缺失题执行有上限的视觉重试。
6. 校验页码、坐标范围、区域冲突、槽位完整性和结果 Schema。
7. 导出题目树、文档 JSON、区域截图、图示资源和槽位可视化。

运行结果固定包含：

- `recognition_mode: "vlm_only"`
- `ocr_used: false`
- `coordinate_authority: "vlm_original_page_pixels"`
- `answer_authority: "vlm_original_page"`

输出仍保留空的 `ocr: []` 字段，便于已有结果消费方平滑升级。项目中没有 OCR 后端或 OCR 降级路径。

![当前 VLM 流程](algorithm_flow.svg)

## 模型配置

当前流程固定使用豆包视觉模型：

```bash
export DOUBAO_API_KEY='your-api-key'
export DOUBAO_BASE_URL='https://ark.cn-beijing.volces.com/api/plan/v3'
export DOUBAO_MODEL='doubao-seed-2.0-lite'
```

正式识别时如果未配置 `DOUBAO_API_KEY`，程序会直接报错。

程序也可从 `~/.config/intelligent-grading-system/doubao.env` 读取密钥。该文件必须仅允许当前用户读取：

```bash
chmod 600 ~/.config/intelligent-grading-system/doubao.env
```

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

## 测试

```bash
python3 -m unittest discover -s tests -v
python3 -m compileall -q exam_pipeline homework_extractor.py
```

测试覆盖全卷视觉构树、页级视觉提取、视觉重试、教师树继承、坐标契约、图示导出和端到端 VLM-only 编排。
