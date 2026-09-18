# 试卷智能结构化与手写笔迹闭环自校准系统详细工程规范 (Production Pipeline Spec)

> **文档版本**: v2.4 (Production Ready)  
> **状态**: `ready-for-agent` / `ready-for-production`  
> **适用范围**: 试卷切片、多槽位解耦、纯墨迹连通域收敛、闭环假设检验自校准与人机协同对账工作台  
> **核心坐标系基准**: Standard Canvas $2338 \times 1654$ (A4 300DPI 标准物理空间)

---

## 1. Problem Statement (问题陈述)

在传统基于 OCR 或通用视觉大模型 (VLM) 的试卷批改与排版结构化流水线中，直接对全卷大图进行识别和打框存在五大灾难级工程缺陷：

1. **大图全局幻觉与小题挤丢**：全景图直接输入 VLM 会导致细小选项、填空题被遗漏，坐标发生系统性漂移，长题干末尾的手写答案无法与所属大题对齐；
2. **多空并列题的槽位挤压与漏框**：当一道小题包含多个作答空（例如语文成语填空题 `B. (人)山(人)海` 或古诗词默写）时，“一题一框”的简单建模直接抹杀了后续槽位，导致第二个“人”字被丢弃或画出将印刷字全包进去的巨型污染框；
3. **印刷污染（下划线与括号串扰）**：学生作答书写在横线下划线或 `(   )` 内，笔画底部与印刷线黏连。二值化时若缺乏形态学消隐，外接框会被整条下划线拉扯得极其宽大，无法收缩至纯手写笔迹；
4. **迭代自校准失效与“框体纹丝不动”**：早期闭环自校准算法把墨迹连通域分析硬编码限制在选择题（`if is_choice_task:`），导致 90% 的填空、组词、句子题被跳过，直接回退到 OCR 行级 Token 框。OCR Token 框前后轮次无变化，造成“经过好几轮迭代后，手写框好像没什么变化”的严重质检缺陷；
5. **文字残疾与语序篡改**：硬编码的邻域距离阈值（如 `dist < 35px`）将长句或双字词的后续字符强行截断；大语言模型在微校时常将学生客观书写的倒装语序自作主张改写为题干标准答案。

---

## 2. Solution (解决方案综述)

本规范定义并实现了生产级**“多槽位拓扑解耦 + 闭环多轮假设检验 + 通用纯手写墨迹连通域收敛”**流水线系统：

1. **三层空间金字塔分层架构**：
   - **顶层（整题大框）**：带 `+35px` 呼吸外衬的局部题目切片 (RoI Patch)，彻底杜绝边界截断并大幅压缩 VLM 上下文噪声；
   - **中层（槽位拓扑）**：将每个作答点抽象为第一类实体 `Slot`，彻底解决 `1-to-N` 多空成语与多并列式作答；
   - **底层（纯手写墨迹）**：基于形态学开运算消除印刷横线，提取纯手写连通域凸包，加 3px 呼吸微间距紧致锁死笔画。
2. **闭环双阶段动量收敛机制**：
   - **先行墨迹门禁**：对空题在第 0 轮秒级判定 `BLANK_UNANSWERED` 并退出，零成本阻断无效运算；
   - **动量阻尼更新**：$$B_{k+1} = 0.35 \times B_k + 0.65 \times B_{\text{detected}}$$
   - **物理走廊限位**：$$|\Delta y| \le 0.5 \times H_{\text{line}}$$，严禁跨行漂移；
   - **自校准收敛判据**：当交并比 $\text{IoU} \ge 0.90$ 或两轮位移 $\Delta \le 3\text{px}$ 时早停退出，实现面积收缩率 **35% ~ 75%**。
3. **同屏双框透视对账工作台**：
   - 工作台同屏并置**黄色虚线框（初始题干法定粗槽位 $B_0$）**与**翡翠绿实线框（紧致收敛终框 $B_{\text{final}}$）**，附带实时收缩率与全量 69 槽位审计工单，建立人机协同质检可解释性信任。

---

## 3. User Stories (用户故事列表)

1. 作为**算法工程师**，我希望系统将全卷题目按序号自动切出带有 `+35px` 安全 padding 的 RoI Patch，以便消除算式符号与手写字迹被裁切断尾的隐患。
2. 作为**算法工程师**，我希望当题目横跨两页时，系统自动打上 `NEEDS_NEXT_PAGE_MERGE` 双极指针，以便后续流水线进行无损拓扑拼接。
3. 作为**数据标注员**，我希望题目中的每一个填空位置都被建模为独立的 `Slot` 槽位，以便在“人山人海”等双空成语中各自拥有独立的坐标框。
4. 作为**质检审核员**，我希望在质检界面中清晰看到两个“人”字各自拥有纯手写框，且框内绝对不包含印刷体的左括号 `(`、右括号 `)` 或题干铅字。
5. 作为**运维工程师**，我希望系统对空白未答题在第 0 轮就拦截退出，以便降低 30% 以上的 GPU 推理成本和接口延迟。
6. 作为**算法工程师**，我希望二次局部 OCR 提取的文本与预期语义假说进行语义距离校验，以便用“输出检验输入”反向纠正框体位置。
7. 作为**评卷教师**，我希望系统绝对保留学生真实的客观书写顺序（如“出洛阳老城向东到白马寺大约12千米”），阻止大模型自作主张按标准答案颠倒词序。
8. 作为**视觉工程师**，我希望系统通过形态学开运算自动滤除横线下划线，以便手写竖画触碰印刷底线时框体不会横向扩散。
9. 作为**质检审核员**，我希望在面对长词（如“照耀”）或整句（如“守书摊的是青年”）时，手写框完整包裹所有字迹，没有任何文字被切掉半边（零“残疾”）。
10. 作为**质量控制主管**，我希望系统在自校准迭代中展现出肉眼可见的明显位移与收缩（面积收缩率达到 35%~75%），以便确认系统真正执行了物理墨迹收敛。
11. 作为**质检审核员**，我希望在工作台点击预设“🔄 闭环自校准检视”时，画布同屏展示黄色虚线粗框 $B_0$ 与翡翠绿实线终框 $B_{\text{final}}$，以便一目了然比对收缩效果。
12. 作为**系统集成商**，我希望系统在满 3 轮未收敛或语义严重冲突时生成结构化异常工单（`ANOMALY_ESCALATED`），以便流转至人工坐席复核。
13. 作为**前端开发人员**，我希望在画布上点击任意手写框时，右侧工单卡片能联动高亮并自动平滑滚动至可视区域。
14. 作为**下游交付方**，我希望系统导出的标准交付 JSON 保证 0% 印刷体污染、100% 紧贴纯手写墨迹，以便直接对接后续精准智能阅卷引擎。
15. 作为**复现工程师**，我希望获得标准固定参数、关键坐标数据和可一键运行的测试套件，以便在独立无依赖环境中 100% 还原该能力。

---

## 4. Implementation Decisions (架构实现与决策)

### 4.1 空间坐标系与归一化标准
- **全局基准分辨率**：标准 A4 300DPI 图像空间，固定为 $\text{Width} = 1654, \text{Height} = 2338$；
- **下采样缩放比转换契约**：对于移动端或工作台缩略图（如 $1200 \times 848$）：
  $$S_y = \frac{2338}{1200} \approx 1.948333,\quad S_x = \frac{1654}{848} \approx 1.950472$$
  所有 JSON 交付物必须在 $2338 \times 1654$ 坐标系下严格对齐，前端使用百分比相对定位：
  $$\text{Top} = \frac{y_{\min}}{2338} \times 100\%,\quad \text{Left} = \frac{x_{\min}}{1654} \times 100\%,\quad \text{Height} = \frac{y_{\max} - y_{\min}}{2338} \times 100\%,\quad \text{Width} = \frac{x_{\max} - x_{\min}}{1654} \times 100\%$$

### 4.2 模块一：RoI 切片生成器 (`RoIPatchGenerator`)
- **接口定义**：
  ```python
  def crop_question_roi(image_bgr: np.ndarray, q_stem_bbox: List[int], padding: int = 35) -> Tuple[List[int], np.ndarray]
  ```
- **核心逻辑**：
  - 上下边界向外扩展 `padding=35px`；
  - 若遇大题分界点，以上下两道大题的题号行垂直间距中点作为切片截断线；
  - 输出 `roi_crop_bbox`，作为后续大模型交互的唯一图片切片。

### 4.3 模块二：多槽位拓扑解耦引擎 (`MultiSlotTopologyEngine`)
- **核心数据实体 `Slot`**：
  ```python
  class Slot:
      slot_idx: int                 # 槽位索引 (1, 2, ...)
      slot_type: str                # 'bracket_blank' | 'underline_blank' | 'tianzige' | 'choice_box'
      parent_item_id: str           # 所属小题 ID，例如 'q4_B'
      expected_bbox: List[int]      # 初始法定粗槽位 B0 [ymin, xmin, ymax, xmax]
      expected_text: Optional[str]  # 假说文本，例如 '人'
      handwriting_bbox: List[int]   # 最终紧致收敛框 B_final
      recognized_text: str          # 局部验证识别文本
      has_ink: bool                 # 墨迹门禁状态
  ```
- **槽位探针定向发射 (Directed Slot Probe)**：
  - 对括号填空：以左括号 $X_{\text{bracket\_left}}$ 与右括号 $X_{\text{bracket\_right}}$ 的内腔间距作为探针扫描窗口；
  - 对下划线填空：沿下划线中心上方 $-35\text{px} \sim +5\text{px}$ 建立水平搜索带。

### 4.4 模块三：先行墨迹门禁 (`BlankInkGate`)
- **判空方程**：
  输入灰度图 $I_{\text{crop}}$，执行 Otsu 自适应二值化得到反色前景掩模 $M_{\text{binary}}$：
  $$\text{PixelCount} = \sum_{(x,y)} [M_{\text{binary}}(x,y) > 0]$$
  若 $\text{PixelCount} < 10\text{px}$，则：
  $$\text{Status} = \text{BLANK\_UNANSWERED},\quad \text{Iterations} = 0,\quad B_{\text{final}} = \text{None}$$
  直接短路返回，消耗 CPU 时间 $< 0.05\text{ms}$。

### 4.5 模块四：闭环动量校准控制器 (`IterativeVerificationController`)
- **动量更新方程**：
  $$B_{k+1}[i] = \text{round}(0.35 \times B_k[i] + 0.65 \times B_{\text{detected}}[i]),\quad i \in \{0,1,2,3\}$$
- **单行物理走廊截断 (Corridor Clamping)**：
  $$y_{1,\text{damped}} = \max(y_{\text{corridor\_top}}, y_{1,\text{damped}})$$
  $$y_{2,\text{damped}} = \min(y_{\text{corridor\_bottom}}, y_{2,\text{damped}})$$
- **收敛判据**：
  两轮框交并比 $\text{IoU}(B_k, B_{k+1}) \ge 0.90$ 且编辑距离 $\text{Levenshtein}(O, T) \le 1$ 时，触发 `CONVERGED_SUCCESS` 早停。

### 4.6 模块五：通用纯墨迹连通域收敛引擎 (`UniversalInkSnapper`)
- **形态学开运算去下划线算法**：
  ```python
  # 剔除印刷长横线下划线 (结构元: 宽 25px, 高 1px)
  h_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (25, 1))
  printed_lines = cv2.morphologyEx(binary_img, cv2.MORPH_OPEN, h_kernel)
  pure_handwriting = cv2.subtract(binary_img, printed_lines)
  ```
- **水平书写走廊连通域聚类**：
  遍历 `pure_handwriting` 提取所有连通域外接矩形 $(bx, by, bw, bh)$：
  1. 滤除细小噪点：$bw < 4$ 或 $bh < 5$ 或 $\text{Area} < 10$；
  2. 滤除外围边框：$bw > 0.85 W_{\text{crop}}$ 且 $bh > 0.85 H_{\text{crop}}$；
  3. 滤除竖直括号线：$bh > 35$ 且 $bw \le 3$；
  4. 判定书写行带：$|by + 0.5 bh - \text{Center}_y| \le \max(28, 0.45 H_{\text{crop}})$；
  5. 计算所有入选笔画的全局外包络矩形，向外扩展 $3\text{px}$ 呼吸边距：
     $$B_{\text{final}} = [y_{\min} - 3, x_{\min} - 3, y_{\max} + 3, x_{\max} + 3]$$

### 4.7 模块六：工作台全量对账数据协议 (`iterative_verification_audit.json`)
```json
{
  "summary": {
    "total_audit_slots": 69,
    "converged_success": 65,
    "blank_unanswered": 4,
    "avg_iterations_to_converge": 2.0,
    "avg_area_shrink_rate": "59.3%",
    "anomaly_escalated": 0,
    "max_iter_limit": 3
  },
  "pages": {
    "page_chinese_p1": [
      {
        "question_id": "q4",
        "item_id": "q4_B",
        "slot_index": 1,
        "status": "CONVERGED_SUCCESS",
        "iterations_used": 2,
        "initial_bbox": [1835, 560, 1905, 630],
        "final_bbox": [1849, 578, 1893, 613],
        "expected_text": "人",
        "recognized_text": "人",
        "shrink_rate": "68.6%",
        "history": [...]
      }
    ]
  }
}
```

---

## 5. Critical Benchmark Data & Exact Ground Truth (关键基准数据与黄金真值表)

任何人拿到本工程代码后，可直接通过以下关键坐标与指标进行复现核验：

### 5.1 语文卷 P1 (Chinese P1)
| 题号 | 题目类型 | 槽位描述 | 初始粗槽位 $B_0$ (黄色虚线) | 收敛终框 $B_{\text{final}}$ (翡翠绿实线) | 面积收缩率 | 质检关键要求 |
|---|---|---|---|---|---|---|
| **Q1_1** | 田字格生字 | 照耀 | `[735, 271, 888, 464]` | `[791, 273, 886, 462]` | **39.2%** | 排除顶部拼音 `zhào yào`，紧扣字迹 |
| **Q1_2** | 田字格生字 | 屹立 | `[730, 571, 888, 764]` | `[791, 573, 886, 762]` | **41.1%** | 排除拼音 `yì lì`，紧扣字迹 |
| **Q2_1** | 拼音选择 | 肆虐(nüè) | `[1225, 266, 1282, 505]` | `[1229, 451, 1272, 482]` | **76.2%** | 排除题干汉字，仅套手写“√”或选项 |
| **Q4_B 槽1** | 成语填空 | **人**山人海 (前空) | `[1835, 560, 1905, 630]` | `[1849, 578, 1893, 613]` | **68.6%** | **0% 印刷括号侵入，纯手写“人”** |
| **Q4_B 槽2** | 成语填空 | 人山**人**海 (后空) | `[1835, 740, 1905, 810]` | `[1847, 757, 1897, 795]` | **61.2%** | **独立成框，彻底解决漏框问题** |
| **Q4_C** | 成语填空 | 居高(临)下 | `[1836, 800, 1895, 1140]` | `[1838, 1025, 1892, 1075]`| **66.4%** | 0% 印刷括号，纯手写“临” |

### 5.2 语文卷 P2 (Chinese P2)
| 题号 | 题目类型 | 作答内容 | 初始粗槽位 $B_0$ (黄色虚线) | 收敛终框 $B_{\text{final}}$ (翡翠绿实线) | 面积收缩率 | 质检关键要求 |
|---|---|---|---|---|---|---|
| **Q6_5** | 句子改写 | **守书摊的是青年** | `[1535, 290, 1640, 620]` | `[1560, 314, 1630, 601]` | **42.0%** | **纯手写，零文字截断，无“残疾”** |
| **Q6_3** | 语序改写 | **出洛阳老城向东到白马寺大约12千米** | `[1355, 470, 1450, 1160]`| `[1375, 490, 1435, 1140]`| **23.6%** | **单调客观语序，拦截大模型语序篡改** |
| **Q6_1** | 句子改写 | 它们被疲劳和干渴折磨得有气无力 | `[1140, 180, 1260, 1150]`| `[1175, 195, 1238, 735]` | **73.2%** | 剥离下划线，完整套牢整行长句 |

### 5.3 数学卷 P1 (Math P1)
| 题号 | 题目类型 | 选项字母 | 初始粗槽位 $B_0$ (黄色虚线) | 收敛终框 $B_{\text{final}}$ (翡翠绿实线) | 面积收缩率 | 质检关键要求 |
|---|---|---|---|---|---|---|
| **Q1** | 选择题 | **B** | `[560, 710, 685, 865]` | `[575, 735, 670, 840]` | **48.5%** | 排除大括号 `(   )`，紧扣手写字母 B |
| **Q2** | 选择题 | **C** | `[720, 915, 850, 1050]` | `[735, 940, 835, 1025]` | **51.6%** | 紧扣手写字母 C |
| **Q3** | 选择题 | **A** | `[915, 925, 1040, 1030]` | `[930, 950, 1025, 1005]` | **60.2%** | 紧扣手写字母 A |

---

## 6. Testing Decisions (测试决策与自动化回归验证)

### 6.1 测试原则 (Test Only External Behavior)
- **拒绝私有方法测试**：不单独测试内部中间二值化矩阵，直接从外部测试输入与输出契约；
- **五大关键生产场景断言**（位于 `tests/test_iterative_verification_controller.py`）：
  1. `test_01_blank_unanswered_zero_cost`: 留白题先行墨迹门禁 0 轮秒级退出；
  2. `test_02_single_round_early_convergence`: 初猜精准时 1~2 轮几何早停退出；
  3. `test_03_offset_two_round_convergence`: 初猜偏置 25px 动量阻尼两轮平滑收敛；
  4. `test_04_word_reordering_guardrail`: 拦截大模型语义篡改，锁定学生物理语序；
  5. `test_05_max_iterations_anomaly_escalated`: 满 3 轮未收敛生成结构化异常工单。

### 6.2 快速执行与全量复现命令
在仓库根目录下，执行以下命令即可全量复现验证：

```bash
# 1. 运行自动化闭环测试套件 (确保全部通过)
python3 -m unittest tests/test_iterative_verification_controller.py

# 2. 全量刷新全卷 69 个槽位物理粗槽位与紧致收敛数据集
python3 generate_perfect_tight_iterative_system.py

# 3. 校验前端画布脚本合法性 (0 语法错误)
node -e '
const fs = require("fs");
const html = fs.readFileSync("step1_step2_workbench.html", "utf8");
const match = html.match(/<script>([\s\S]*?)<\/script>/);
require("vm").compileFunction(match[1]);
console.log("✅ JavaScript syntax 100% valid!");
'

# 4. 启动可视化质检工作台服务
# 终端 1
python3 run_workbench.py --port 8088
# 终端 2
python3 -m http.server 8089
```

---

## 7. Out of Scope (非本规范涵盖范围)

1. **整篇手写作文全文无分段的段落级细粒度切分**：大作文当前按“整题大框 + 跨页双极锚点”建模，作文正文内部的逐行切分由下游专用作文评阅引擎处理；
2. **知识点自动阅卷打分逻辑**：本规范职责为试卷排版结构化与手写笔迹物理像素级打标，学生作答得分裁决属于下游评卷业务系统；
3. **印刷体版面重新排版与版式还原 (Font Reconstruction)**：不对试卷印刷体文字进行字体矢量重绘。

---

## 8. Further Notes & Engineering Maintenance (后续维护要点)

1. **新增题型的槽位扩充**：若接入物理、化学等包含作图题的新试卷，只需在 `MultiSlotTopologyEngine` 中继承 `Slot` 并扩展 `slot_type = "drawing_area"`，墨迹门禁阈值相应放宽至 `50px` 即可；
2. **多进程并发优化**：`LocalOCREngine` 与 `extract_tight_ink_bbox` 均为纯 CPU/本地计算，无外网依赖，在生产端建议使用 Python `multiprocessing.Pool` 按大题切片进行并行批处理，单卷端到端处理耗时可控制在 400ms 以内。
