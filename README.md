# 智能试卷识别提取系统

基于 OCR 和多模态模型的试卷自动识别提取系统，支持从教师卷和学生卷扫描图中提取题目结构和答题内容。

## 核心功能

- **试卷结构识别**：自动识别题目编号、题干、答题区域
- **教师答案提取**：从教师卷中提取标准答案
- **学生答案提取**：从学生卷中提取手写答案
- **墨迹分离**：通过 cohort consensus 方法分离印刷内容和手写内容
- **迭代验证**：多轮迭代精确定位答题区域
- **结构化输出**：生成 JSON 格式的提取结果

## 项目结构

```
intelligent-grading-system/
├── homework_extractor.py          # 主识别程序（唯一入口）
├── exam_pipeline/                 # 核心识别算法模块
│   ├── service.py                # 流程服务主控制器
│   ├── ocr_service.py            # OCR 识别服务
│   ├── registration.py           # 图像配准（对齐）
│   ├── slots.py                  # 答题槽位检测
│   ├── roi.py                    # 答题区域提取
│   ├── teacher_answers.py        # 教师答案提取
│   ├── cohort_consensus.py       # 群体共识墨迹分离
│   ├── verification.py           # 迭代验证算法
│   └── ...                       # 其他核心模块
├── GROUNDING_KNOWLEDGE_BASE.md    # 迭代验证算法规则库
├── requirements.txt               # Python 依赖
├── .env.example                   # 环境配置示例
├── 扫描图片/                      # 测试数据
└── test_dataset/                  # 测试数据集
```

## 快速开始

### 1. 安装依赖

```bash
pip install -r requirements.txt
```

### 2. 准备数据

将试卷扫描图按以下结构组织：

```
数据目录/
└── 科目名/
    ├── teacher/          # 教师卷（2页）
    │   ├── 扫描作业_001.jpg
    │   └── 扫描作业_002.jpg
    ├── student001/       # 学生1（2页）
    │   ├── 扫描作业_003.jpg
    │   └── 扫描作业_004.jpg
    ├── student002/       # 学生2（2页）
    │   └── ...
    └── ...
```

### 3. 运行识别

**基本用法：**
```bash
python3 homework_extractor.py 数据目录 \
  --subject 科目名 \
  --ocr paddle \
  -o 输出目录
```

**示例：**
```bash
python3 homework_extractor.py 扫描图片 \
  --subject physics_0105 \
  --ocr paddle \
  -o extracted_results
```

**试运行（仅检查数据分组，不调用 OCR）：**
```bash
python3 homework_extractor.py 扫描图片 \
  --subject physics_0105 \
  --dry-run \
  -o test_check
```

### 4. 查看结果

识别结果保存在输出目录中：

```
输出目录/
├── manifest.json                      # 总体清单
├── 科目名__teacher__teacher.json      # 教师卷提取结果
├── 科目名__student__student001.json   # 学生卷提取结果
├── exam_trees/                        # 题目树
│   └── 科目名.json
├── normalized/                        # 归一化后的图片
├── roi_patches/                       # 答题区域截图
└── cohort_consensus/                  # 共识模板
```

## 主要参数说明

### OCR 引擎
- `--ocr paddle`：使用 PaddleOCR（推荐）
- `--ocr tesseract`：使用 Tesseract OCR
- `--ocr none`：跳过 OCR，使用预生成的 OCR 结果

### 图像处理
- `--paddle-model-tier mobile`：使用轻量级模型（默认）
- `--paddle-model-tier server`：使用服务器级模型（更准确）

### 共识墨迹分离
- `--consensus-max-samples 10`：使用最多 10 个学生样本生成共识模板
- `--save-separation-masks`：保存墨迹分离的中间结果

### 调试选项
- `--dry-run`：仅扫描分组，不调用 OCR/VLM
- `--limit N`：只处理前 N 份试卷
- `--student-id student001`：只处理指定学生

## 算法流程

![算法流程图](algorithm_flow.svg)

1. **图像预处理**：页面归一化到 1654×2338，透视校正
2. **OCR 识别**：PP-OCRv5 整页识别，定位题目和文本
3. **结构分析**：识别题目编号、题干、答题区域
4. **Cohort Consensus**：多份学生卷对齐投票，生成印刷模板
5. **墨迹分离**：从学生卷中分离手写内容
6. **教师答案提取**：提取教师卷的标准答案
7. **题目树生成**：编译、校验、锁定 Exam Tree
8. **学生答案提取**：基于题目树提取每个学生的答案
9. **迭代验证**：最多三轮动量收敛，精确定位答题区域
10. **质量评估**：生成审计日志和质量报告

## 迭代验证算法

系统使用知识库驱动的迭代验证算法（配置在 [GROUNDING_KNOWLEDGE_BASE.md](GROUNDING_KNOWLEDGE_BASE.md)）：

- **墨迹中心定位**：向墨迹中心收缩
- **笔画安全边距**：保护完整笔画
- **多行文本合并**：合并多行答案
- **印刷内容隔离**：避免包含印刷题干
- **列隔离**：避免跨列提取
- **墨迹存在门控**：检测是否有手写内容
- **模板差分**：使用教师卷作为模板

## 测试数据

项目包含两套测试数据：

**扫描图片/**
- `physics_0105/`：物理试卷（1 教师卷 + 10 学生卷）
- `math_01/`：数学试卷

**test_dataset/**
- 额外的测试样本

## 输出 JSON 格式

```json
{
  "exam_id": "physics_0105__student__student001",
  "subject": "physics_0105",
  "document_type": "student",
  "student_id": "student001",
  "sections": [
    {
      "section_id": "sec_choice",
      "section_title": "一、选择题",
      "questions": [
        {
          "question_id": "q1",
          "items": [
            {
              "item_id": "q1",
              "standard_answer": "C",
              "student_answer": "C",
              "answer_regions": [...],
              "slots": [...]
            }
          ]
        }
      ]
    }
  ]
}
```

## 环境配置

复制 `.env.example` 为 `.env` 并配置：

```bash
# 豆包 API（用于语义结构理解）
DOUBAO_API_KEY=your-api-key
DOUBAO_BASE_URL=https://ark.cn-beijing.volces.com/api/plan/v3
DOUBAO_MODEL=doubao-seed-2.0-lite

# OCR 配置
EXAM_OCR_ENGINE=paddle
EXAM_OCR_LANGUAGE=chi_sim+eng
EXAM_LOCAL_OCR=true

# PaddleOCR 缓存
PADDLE_PDX_CACHE_HOME=.paddlex-cache
```

## 常见问题

### Q: 为什么有些学生答案提取失败？
A: 可能原因：
- 学生答题位置偏离模板太大
- 手写内容太淡或太潦草
- 墨迹分离效果不佳

解决方法：
- 增加共识样本数量（`--consensus-max-samples`）
- 使用更高质量的扫描图
- 检查 `iterative_verification_audit.json` 查看详细失败原因

### Q: 如何提高识别准确率？
A: 建议：
- 使用更多学生样本生成共识模板（至少 3-5 份）
- 确保扫描图清晰、对比度高
- 学生答题位置尽量规范
- 优化 `GROUNDING_KNOWLEDGE_BASE.md` 中的规则

### Q: 如何调试识别问题？
A: 步骤：
1. 使用 `--save-separation-masks` 保存中间结果
2. 查看 `iterative_verification_audit.json` 了解每个槽位的识别状态
3. 检查 `roi_patches/` 中的答题区域截图
4. 查看 `manifest.json` 中的质量指标

## 技术特点

- **无需空白卷**：通过 cohort consensus 自动生成印刷模板
- **鲁棒性强**：支持倾斜、透视、光照变化的扫描图
- **高精度定位**：迭代验证算法，IOU > 0.9
- **可解释性**：完整的审计日志和质量报告
- **模块化设计**：核心算法独立于 CLI，便于集成

## 技术栈

- **OCR 引擎**：PaddleOCR v5 (mobile/server)
- **图像处理**：OpenCV, PIL
- **多模态模型**：豆包 VLM（可选）
- **图像配准**：ECC Affine
- **语言**：Python 3.8+

## 许可证

本项目仅供学习和研究使用。

## 相关文档

- [GROUNDING_KNOWLEDGE_BASE.md](GROUNDING_KNOWLEDGE_BASE.md) - 迭代验证算法规则库
- [PRODUCTION_PIPELINE_SPEC.md](PRODUCTION_PIPELINE_SPEC.md) - 生产流程技术规范

## 联系方式

如有问题或建议，请提交 Issue。
