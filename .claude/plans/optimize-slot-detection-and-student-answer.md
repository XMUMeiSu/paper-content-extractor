# 优化填空题槽位检测和学生答案精细化方案

## 问题诊断

### 当前状态（extracted_math_01_latest）

#### ✅ 正确的部分
1. **教师卷题目提取**：准确
2. **教师卷答案提取**：准确  
3. **学生区域独立检测**：已启用（`student_independent_detection`标记）
4. **Item级别student_answer**：已提取（由`GeometryGrounder`完成）

#### ❌ 存在的问题

**问题1：Slot级别的OCR识别包含大量噪音**
```
recognized_text: "1.在平面直角坐标系中，二次函数y=(x+1)²的图象可能是 yA"
```
- 应该只识别学生手写答案（如"C"），但包含了题目、题号等印刷文本
- 墨迹分离（ink separation）没有正确隔离手写内容

**问题2：填空题槽位定位不准确**
- 数学卷Q4-Q5：`localized_slot_count: 0`（应该≥1）
- 数学卷Q6：`localized_slot_count: 2`（应该是5）
- 物理卷类似问题

**问题3：学生答案区域仍然过大**
```
bbox: [127.0, 456.0, 1625.0, 613.0]  // 宽1498px，包含整个题目区域
```

---

## 优化方案

### 第一阶段：增强填空题槽位检测

#### 1.1 改进横线/下划线检测
**文件**：`exam_pipeline/slots.py`中的`_underline_slots`

**问题**：当前依赖OpenCV的线检测，容易漏检手写填充的下划线

**优化**：
- 添加OCR文本模式检测（`_{2,}`, `\.{4,}`等）
- 结合layout检测的水平rule元素
- 增加连续空白区域检测（印刷题目中的答题空间）

#### 1.2 改进括号/方框槽位检测
**文件**：`exam_pipeline/slots.py`中的`_bracket_slots`

**问题**：只检测简单的括号对，漏掉`【】`、`〔〕`等变体

**优化**：
- 扩展括号正则表达式
- 添加空方框检测（`□`、空白框线）
- 检测"第___空"、"答：___"等模式

#### 1.3 增强多空题的slot count推断
**文件**：`exam_pipeline/cardinality.py`

**当前逻辑**：
```python
def _expected_slot_count(self, item: ExamItem) -> int:
    # 依赖teacher answer结构
    if isinstance(answer, dict): return len(answer)
    if isinstance(answer, list): return len(answer)
```

**优化**：
- 从题目文本中提取序号标记（"(1)"、"①"、"第一空"等）
- 统计下划线、括号数量作为fallback
- 增加VLM视觉计数（已有但未充分利用）

---

### 第二阶段：学生答案精细化提取

#### 2.1 改进墨迹分离算法
**文件**：`exam_pipeline/verification.py`中的`InkSeparationController`

**当前问题**：residual方法仍然包含印刷文本

**优化策略**：
1. **增强模板对齐**：
   - 使用ORB/SIFT特征点匹配，而不是简单的像素差分
   - 对学生卷进行仿射变换，精确对齐到教师模板
   
2. **多层次过滤**：
   ```
   Layer 1: 模板差分（去除大部分印刷）
   Layer 2: 笔画粗细过滤（手写通常比印刷粗/细）
   Layer 3: 颜色过滤（蓝/黑笔vs印刷黑）
   Layer 4: 连通域分析（过滤小噪点）
   ```

3. **边界收缩**：
   - 当前`handwriting_bbox`仍然过大
   - 在墨迹mask基础上做morphology收缩
   - 计算最小外接矩形

#### 2.2 选择题答案提取专项优化
**文件**：新建`exam_pipeline/choice_answer_extractor.py`

**问题**：选择题学生只需勾选/圈选选项，但OCR识别了整个区域

**方案**：
1. **圆圈检测**：
   - Hough圆检测找到学生圈选的选项
   - 判断圆内是否有字母（A/B/C/D）
   
2. **勾选标记检测**：
   - 检测✓、√、×等符号
   - 基于位置映射到选项

3. **涂卡识别**：
   - 对于标准化答题卡，检测涂黑区域
   - 计算各选项区域的黑色像素密度

#### 2.3 填空题答案OCR专项优化
**文件**：`exam_pipeline/grounding.py`中的`GeometryGrounder.refine_item`

**当前问题**：
- padding过大（28-35px），包含周围印刷文本
- 没有按slot分别裁剪，而是整个区域OCR

**优化**：
1. **按槽位独立OCR**：
   ```python
   for slot in item.slots:
       # 只OCR slot.expected_bbox + 小padding(5-10px)
       slot_blocks = ocr.recognize_crop(page, slot.expected_bbox, padding=8)
       slot.recognized_text = filter_handwriting_only(slot_blocks)
   ```

2. **手写过滤**：
   - 先做墨迹分离得到mask
   - 只保留mask内的OCR结果
   - 过滤掉与题目文本相似度高的内容

---

### 第三阶段：答案语义提取

#### 3.1 从recognized_text提取纯净答案
**文件**：新建`exam_pipeline/answer_parser.py`

**功能**：从混合文本中提取真实学生答案

**策略**：
1. **选择题**：
   - 正则匹配A/B/C/D（忽略题目中的选项标记）
   - 优先使用靠近slot bbox的字符
   
2. **填空题**：
   - 如果`recognized_text`包含`expected_text`→可能正确
   - 否则提取数字/公式/短语（排除长句子）
   
3. **添加`student_answer`字段到Slot**：
   修改`exam_pipeline/contracts.py`:
   ```python
   @dataclass
   class Slot:
       ...
       recognized_text: str = ""  # OCR原始结果
       student_answer: Optional[str] = None  # 提取的纯净答案
   ```

#### 3.2 答案标准化
- 去除空格、标点
- 数字格式统一（1.0 → 1）
- 数学符号归一化（×→*，÷→/）

---

## 实施计划

### Step 1: 修复contracts.py，添加student_answer字段
```python
# exam_pipeline/contracts.py
@dataclass
class Slot:
    ...
    student_answer: Optional[str] = None  # 新增字段
```

### Step 2: 创建answer_parser.py
实现从recognized_text中提取纯净答案的逻辑

### Step 3: 修改verification.py
在verification完成后调用answer_parser

### Step 4: 优化cardinality.py的槽位计数
增强多空题的slot_count推断

### Step 5: 优化slots.py的槽位检测
改进下划线、括号检测

### Step 6: 优化grounding.py的按slot OCR
每个slot独立裁剪+OCR

### Step 7: 增强墨迹分离
改进特征点匹配和多层过滤

---

## 验证标准

提取完成后，检查：

### 数学卷验证点
- [ ] Q1-Q3（选择题）：`slot.student_answer`只包含选项字母（A/B/C/D），无题目文本
- [ ] Q4-Q5（单空填空）：`localized_slot_count >= 1`，答案是纯数字/公式
- [ ] Q6（5空填空）：`localized_slot_count = 5`，每个slot有独立答案

### 物理卷验证点
- [ ] 选择题：答案干净
- [ ] 填空题：所有空都能定位
- [ ] 实验题：每个子问有独立槽位

### 通用质量指标
- [ ] `recognized_text`长度 < 50字符（除大题外）
- [ ] `student_answer`不包含题号、题目关键词
- [ ] `geometry_status: ALIGNED`比例 > 80%
- [ ] `content_status: RECOGNIZED`比例 > 70%

---

## 时间估算
- Step 1-3（核心答案提取）：2小时
- Step 4-5（槽位检测优化）：3小时  
- Step 6-7（OCR和墨迹优化）：3小时
- 测试和调优：2小时
- **总计**：10小时
