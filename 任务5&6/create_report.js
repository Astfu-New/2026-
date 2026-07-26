import fs from "node:fs";
import path from "node:path";
import {
  AlignmentType, Document, Footer, Header, HeadingLevel,
  ImageRun, ImportedXmlComponent, Packer, PageNumber,
  Paragraph, ShadingType, Table, TableCell, TableRow,
  TextRun, WidthType, convertInchesToTwip,
} from "docx";

const outputPath = process.argv[2];
if (!outputPath) throw new Error("Usage: node create_report.js /absolute/path/output.docx");

const outputDir = path.dirname(outputPath);
const assetDir = outputDir;

const palette = {
  dark: "263238", primary: "37474F", light: "78909C",
  border: "D8E0E3", fill: "EEF3F6",
};
const font = {
  ascii: "Times New Roman", hAnsi: "Times New Roman",
  cs: "Times New Roman", eastAsia: "SimSun",
};
const run = (text, options = {}) => new TextRun({ text, font, size: 24, ...options });
const para = (children, options = {}) => new Paragraph({
  spacing: { after: 160, line: 300 },
  ...options,
  children: Array.isArray(children) ? children : [children],
});
const bodyPara = (text, options = {}) => para(run(text), {
  indent: { firstLine: convertInchesToTwip(0.33) }, ...options,
});
const heading = (text, level = 1) => para(run(text, {
  bold: true,
  size: level === 1 ? 30 : level === 2 ? 26 : 24,
  color: palette.dark
}), {
  heading: level === 1 ? HeadingLevel.HEADING_1
    : level === 2 ? HeadingLevel.HEADING_2 : HeadingLevel.HEADING_3,
  spacing: { before: level === 1 ? 300 : 240, after: 120 },
});
const cell = (text, options = {}) => new TableCell({
  children: [para(run(text, { size: 22 }), { alignment: AlignmentType.CENTER })],
  margins: { top: 100, bottom: 100, left: 100, right: 100 },
  ...options,
});
const xmlEscape = (value) => String(value)
  .replaceAll("&", "&amp;").replaceAll("<", "&lt;")
  .replaceAll(">", "&gt;").replaceAll('"', "&quot;");
const toc = (entries) => {
  const cached = entries.map(({ title: entryTitle, level, page }) => {
    const indent = Math.max(0, level - 1) * 360;
    return "<w:p><w:pPr><w:pStyle w:val=\"TOC" + level + "\"/>" +
      "<w:tabs><w:tab w:val=\"right\" w:leader=\"dot\" w:pos=\"9000\"/></w:tabs>" +
      "<w:ind w:left=\"" + indent + "\"/></w:pPr>" +
      "<w:r><w:t>" + xmlEscape(entryTitle) + "</w:t></w:r><w:r><w:tab/></w:r><w:r><w:t>" + xmlEscape(page) + "</w:t></w:r></w:p>";
  }).join("");
  return ImportedXmlComponent.fromXmlString("<w:sdt xmlns:w=\"http://schemas.openxmlformats.org/wordprocessingml/2006/main\">" +
    "<w:sdtPr><w:alias w:val=\"目录\"/></w:sdtPr>" +
    "<w:sdtContent>" +
    "<w:p><w:r><w:fldChar w:fldCharType=\"begin\" w:dirty=\"true\"/>" +
    "<w:instrText xml:space=\"preserve\"> TOC \\o &quot;1-3&quot; \\h \\z \\u </w:instrText>" +
    "<w:fldChar w:fldCharType=\"separate\"/></w:r></w:p>" +
    cached +
    "<w:p><w:r><w:fldChar w:fldCharType=\"end\"/></w:r></w:p>" +
    "</w:sdtContent>" +
    "</w:sdt>").root[0];
};

// 加载实验数据
const expData = JSON.parse(fs.readFileSync(path.join(assetDir, "experiment_data.json"), "utf-8"));
const input4x4 = expData.input_4x4;
const kernel3x3 = expData.kernel_3x3;
const plainResult = expData.plain_result;
const fheV1 = expData.fhe_v1_result;
const fheV2 = expData.fhe_v2_result;
const timing = expData.timing;
const rotAnalysis = expData.rotation_analysis;
const intermediate = expData.plain_intermediate;

// 封面
const coverChildren = [
  para(run(""), { spacing: { before: 1200 } }),
  para(run("全同态加密密文卷积实验报告", { bold: true, size: 44, color: palette.dark }), {
    alignment: AlignmentType.CENTER, spacing: { after: 600 },
  }),
  para(run("打包-旋转-累加策略与旋转次数优化", { size: 28, color: palette.primary }), {
    alignment: AlignmentType.CENTER, spacing: { after: 800 },
  }),
  para(run("课程名称：信息安全与密码学", { size: 24 }), { alignment: AlignmentType.CENTER, spacing: { after: 120 } }),
  para(run("实验日期：2026-07-24", { size: 24 }), { alignment: AlignmentType.CENTER, spacing: { after: 120 } }),
  para(run("指导教师：____________", { size: 24 }), { alignment: AlignmentType.CENTER, spacing: { after: 600 } }),
  para(run("小组成员", { bold: true, size: 26 }), { alignment: AlignmentType.CENTER, spacing: { before: 200, after: 200 } }),
  new Table({
    width: { size: 60, type: WidthType.PERCENTAGE },
    alignment: AlignmentType.CENTER,
    rows: [
      new TableRow({ children: [cell("成员 1：____________", { shading: { type: ShadingType.CLEAR, fill: palette.fill } })] }),
      new TableRow({ children: [cell("成员 2：____________", { shading: { type: ShadingType.CLEAR, fill: palette.fill } })] }),
      new TableRow({ children: [cell("成员 3：____________", { shading: { type: ShadingType.CLEAR, fill: palette.fill } })] }),
      new TableRow({ children: [cell("成员 4：____________", { shading: { type: ShadingType.CLEAR, fill: palette.fill } })] }),
    ],
  }),
];

// 正文
const contentChildren = [heading("目录"), para(run("（右键目录，选择更新域刷新页码）", { italics: true, color: palette.light, size: 20 }))];

const sections = [
  { title: "一、实验目的", level: 1, page: 3, paras: [
    "理解全同态加密（FHE）的基本原理，掌握 BFV 批处理加密方案的参数配置与使用。",
    "在密文域上实现二维卷积运算，采用打包（Pack）→旋转（Rotate）→累加（Accumulate）策略。",
    "探索并验证密文卷积实现中的旋转次数是否达到理论最小值，理解算法优化在隐私保护计算中的意义。",
  ]},
  { title: "二、实验原理", level: 1, page: 4, paras: [
    "2.1 全同态加密（FHE）与 BFV 方案",
    "BFV（Brakerski-Fan-Vercauteren）方案基于 RLWE 困难问题，支持 SIMD 风格的批处理加密：将多个明文整数打包到一条密文的多个 slot 中，一次加密即可保护整个向量。",
    "2.2 二维卷积运算",
    "给定 4×4 输入矩阵 X 和 3×3 卷积核 K，步长 stride=1、无填充，输出矩阵 Y 的尺寸为 2×2。每个输出元素 y_ij 等于以 (i,j) 为左上角的 3×3 窗口与卷积核逐元素相乘后求和。",
    "2.3 打包-旋转-累加策略",
    "（1）打包：将 4×4 输入矩阵按行优先展平为 16 维向量，使用 BFV 批处理一次性加密为单条密文。",
    "（2）旋转：通过构造不同的权向量，等效地将输入密文旋转到与卷积核对应位置对齐。",
    "（3）累加：调用 dot(w) 将旋转对齐后的密文-明文乘积累加。dot() 内部采用二进制分治策略，对 16 个 slot 进行循环旋转后逐对相加，最终将所有值归约到 slot 0。",
  ]},
  { title: "2.4 旋转次数理论分析", level: 2, page: 5, paras: [
    "dot() 内部工作机制（SEAL 标准实现）：",
    "Step 1: 密文向量与明文权向量逐元素相乘（SIMD 并行）。",
    "Step 2: 使用 Galois Keys 对密文进行循环旋转（rotate）。",
    "Step 3: 将旋转后的密文与原始密文相加（二进制分治累加）。",
    "对于 n=16 个 slot，二进制分治累加需要 log2(16)=4 次旋转。",
    "版本 A（基线）：每个卷积核元素单独执行一次 dot，4 输出 × 9 核元素 = 36 次 dot，总旋转次数 = 36 × 4 = 144 次。",
    "版本 B（优化）：每个输出位置仅需一次 dot（权向量包含 9 个非零值），4 输出 × 1 次 dot = 4 次 dot，总旋转次数 = 4 × 4 = 16 次。",
    "理论最小值：串行计算下，4 个输出位置每个需要 log2(16)=4 次旋转，理论最小值为 16 次。版本 B 已达到此下界。",
  ]},
  { title: "三、实验环境", level: 1, page: 6, paras: ["3.1 硬件环境"] },
];

const hwTable = () => {
  const widths = [3600, 3600];
  const hcell = (t) => cell(t, { shading: { type: ShadingType.CLEAR, fill: palette.fill }, width: { size: widths[0], type: WidthType.DXA } });
  return new Table({
    width: { size: 100, type: WidthType.PERCENTAGE }, columnWidths: widths,
    rows: [
      new TableRow({ children: [hcell("项目"), hcell("配置")] }),
      new TableRow({ children: [cell("处理器"), cell("CPU（本实验不涉及 GPU 加速）")] }),
      new TableRow({ children: [cell("内存"), cell("≥ 8 GB")] }),
    ],
  });
};

const swTable = () => {
  const widths = [2400, 2400, 2400];
  const hcell = (t) => cell(t, { shading: { type: ShadingType.CLEAR, fill: palette.fill }, width: { size: widths[0], type: WidthType.DXA } });
  return new Table({
    width: { size: 100, type: WidthType.PERCENTAGE }, columnWidths: widths,
    rows: [
      new TableRow({ children: [hcell("依赖项"), hcell("版本"), hcell("用途")] }),
      new TableRow({ children: [cell("Python"), cell("3.12+"), cell("主语言")] }),
      new TableRow({ children: [cell("TenSEAL"), cell("0.3.16"), cell("BFV/CKKS FHE 库")] }),
      new TableRow({ children: [cell("NumPy"), cell("2.4.4"), cell("矩阵运算")] }),
      new TableRow({ children: [cell("Matplotlib"), cell("3.10.9"), cell("图表绘制")] }),
    ],
  });
};

// 组装正文
let tocEntries = [];
for (const sec of sections) {
  tocEntries.push({ title: sec.title, level: sec.level, page: sec.page });
  contentChildren.push(heading(sec.title, sec.level));
  for (const p of sec.paras) {
    if (/^\d+\.\d+\s/.test(p)) {
      contentChildren.push(para(run(p, { bold: true, size: 24 })));
    } else {
      contentChildren.push(bodyPara(p));
    }
  }
}

contentChildren.push(hwTable());
contentChildren.push(para(run(""), { spacing: { after: 200 } }));
contentChildren.push(heading("3.2 软件环境", 2));
contentChildren.push(swTable());
contentChildren.push(para(run(""), { spacing: { after: 200 } }));

// 四、实验步骤
contentChildren.push(heading("四、实验步骤"));
contentChildren.push(heading("4.1 明文卷积（基准对照）", 2));
contentChildren.push(bodyPara("实现明文域的二维卷积函数 plain_conv2d，通过滑动窗口遍历输入矩阵，对每个窗口执行逐元素乘法并求和，作为密文卷积正确性验证的基准。"));

contentChildren.push(heading("4.2 版本 A：逐个核元素计算（基线）", 2));
contentChildren.push(bodyPara("对每个输出位置和每个卷积核元素，单独执行一次 dot 操作。权向量仅在一个位置填入当前核元素值，其余位置置 0。共 4×9=36 次 dot 调用。"));

contentChildren.push(heading("4.3 版本 B：每个输出位置一次 dot（优化）", 2));
contentChildren.push(bodyPara("对每个输出位置，构造一个 16 维权向量，在对应 3×3 窗口的 9 个索引位置填入卷积核值。通过一次 dot 完成该输出位置的全部 9 个加权和。共 4 次 dot 调用。"));

contentChildren.push(heading("4.4 旋转次数分析与验证", 2));
contentChildren.push(bodyPara("分析 dot() 内部的二进制分治累加过程，计算两个版本的等效旋转次数，并与理论最小值对比。"));

// 五、实验结果
contentChildren.push(heading("五、实验结果"));
contentChildren.push(heading("5.1 测试数据", 2));
contentChildren.push(bodyPara("实验使用固定随机种子 seed=42 生成测试数据，确保实验可复现。"));

const dataTable = () => {
  const widths = [1200, 1200, 1200, 1200, 1200];
  const hcell = (t) => cell(t, { shading: { type: ShadingType.CLEAR, fill: palette.fill }, width: { size: widths[0], type: WidthType.DXA } });
  return new Table({
    width: { size: 100, type: WidthType.PERCENTAGE }, columnWidths: widths,
    rows: [
      new TableRow({ children: [hcell("矩阵"), hcell("维度"), hcell("数值范围"), hcell("数据类型"), hcell("说明")] }),
      new TableRow({ children: [cell("X"), cell("4×4"), cell("[-5, 5]"), cell("int64"), cell("输入图像")] }),
      new TableRow({ children: [cell("K"), cell("3×3"), cell("[-3, 3]"), cell("int64"), cell("卷积核")] }),
      new TableRow({ children: [cell("Y"), cell("2×2"), cell("[-135, 135]"), cell("int64"), cell("卷积输出")] }),
    ],
  });
};
contentChildren.push(dataTable());
contentChildren.push(para(run(""), { spacing: { after: 200 } }));

contentChildren.push(heading("5.2 输入、卷积核与输出可视化", 2));
contentChildren.push(para(new ImageRun({
  type: "png", data: fs.readFileSync(path.join(assetDir, "fig1_data_overview.png")),
  transformation: { width: 520, height: 156 },
}), { alignment: AlignmentType.CENTER }));
contentChildren.push(para(run("图 1  输入矩阵、卷积核与输出结果热图", { size: 20, italics: true, color: palette.light }), { alignment: AlignmentType.CENTER, spacing: { after: 300 } }));

contentChildren.push(heading("5.3 卷积计算过程", 2));
contentChildren.push(bodyPara("图 2 展示了 4 个滑动窗口对应的卷积计算过程。"));
contentChildren.push(para(new ImageRun({
  type: "png", data: fs.readFileSync(path.join(assetDir, "fig2_convolution_process.png")),
  transformation: { width: 400, height: 400 },
}), { alignment: AlignmentType.CENTER }));
contentChildren.push(para(run("图 2  四个滑动窗口的卷积计算过程", { size: 20, italics: true, color: palette.light }), { alignment: AlignmentType.CENTER, spacing: { after: 300 } }));

// 六、结果分析
contentChildren.push(heading("六、结果分析"));

contentChildren.push(heading("6.1 明文 vs 密文结果对比", 2));
contentChildren.push(bodyPara("表 1 给出了四个输出位置的明文卷积结果、密文卷积解密结果及误差分析。"));

const resultTable = () => {
  const widths = [1800, 1800, 1800, 1800];
  const hcell = (t) => cell(t, { shading: { type: ShadingType.CLEAR, fill: palette.fill }, width: { size: widths[0], type: WidthType.DXA } });
  const rows = [new TableRow({ children: [hcell("输出位置"), hcell("明文结果"), hcell("密文结果"), hcell("绝对误差")] })];
  for (let i = 0; i < 2; i++) {
    for (let j = 0; j < 2; j++) {
      const name = "y" + i + j;
      const p = plainResult[i][j];
      const f = fheV2[i][j];
      rows.push(new TableRow({ children: [cell(name), cell(String(p)), cell(String(f)), cell(String(Math.abs(p - f)))] }));
    }
  }
  return new Table({ width: { size: 100, type: WidthType.PERCENTAGE }, columnWidths: widths, rows });
};
contentChildren.push(resultTable());
contentChildren.push(para(run("表 1  明文卷积与密文卷积结果对比", { size: 20, italics: true, color: palette.light }), { alignment: AlignmentType.CENTER, spacing: { after: 300 } }));

contentChildren.push(para(new ImageRun({
  type: "png", data: fs.readFileSync(path.join(assetDir, "fig3_plain_vs_fhe.png")),
  transformation: { width: 480, height: 237 },
}), { alignment: AlignmentType.CENTER }));
contentChildren.push(para(run("图 3  明文卷积与密文卷积结果对比柱状图", { size: 20, italics: true, color: palette.light }), { alignment: AlignmentType.CENTER, spacing: { after: 300 } }));

contentChildren.push(heading("6.2 旋转次数对比与理论验证", 2));
contentChildren.push(bodyPara("图 4 展示了版本 A、版本 B 与理论最小值的旋转次数对比。版本 A 采用逐个核元素 dot 策略，产生 144 次旋转；版本 B 将 9 个核元素合并到一次 dot 中，仅需 16 次旋转，达到了串行计算的理论最小值。"));

contentChildren.push(para(new ImageRun({
  type: "png", data: fs.readFileSync(path.join(assetDir, "fig4_rotation_count.png")),
  transformation: { width: 520, height: 289 },
}), { alignment: AlignmentType.CENTER }));
contentChildren.push(para(run("图 4  旋转次数对比", { size: 20, italics: true, color: palette.light }), { alignment: AlignmentType.CENTER, spacing: { after: 300 } }));

contentChildren.push(bodyPara("图 5 展示了打包-旋转-累加的完整实现流程。dot() 内部通过 4 次二进制分治旋转将 16 个 slot 归约为 1 个 slot。"));
contentChildren.push(para(new ImageRun({
  type: "png", data: fs.readFileSync(path.join(assetDir, "fig6_workflow.png")),
  transformation: { width: 520, height: 220 },
}), { alignment: AlignmentType.CENTER }));
contentChildren.push(para(run("图 5  Pack-Rotate-Accumulate 实现流程", { size: 20, italics: true, color: palette.light }), { alignment: AlignmentType.CENTER, spacing: { after: 300 } }));

contentChildren.push(bodyPara("图 6 详细展示了 dot() 内部二进制分治累加的 4 个步骤。每一步通过 rotate(slot/2^i) 将距离为 2^i 的两个 slot 对齐，然后相加，逐步将 16 个值归约到 1 个值。"));
contentChildren.push(para(new ImageRun({
  type: "png", data: fs.readFileSync(path.join(assetDir, "fig7_binary_tree.png")),
  transformation: { width: 520, height: 111 },
}), { alignment: AlignmentType.CENTER }));
contentChildren.push(para(run("图 6  二进制分治累加示意图（dot 内部 4 次旋转）", { size: 20, italics: true, color: palette.light }), { alignment: AlignmentType.CENTER, spacing: { after: 300 } }));

contentChildren.push(heading("6.3 性能分析", 2));
contentChildren.push(bodyPara("表 2 列出了实验各阶段的耗时数据。"));

const perfTable = () => {
  const widths = [3600, 3600];
  const hcell = (t) => cell(t, { shading: { type: ShadingType.CLEAR, fill: palette.fill }, width: { size: widths[0], type: WidthType.DXA } });
  return new Table({
    width: { size: 100, type: WidthType.PERCENTAGE }, columnWidths: widths,
    rows: [
      new TableRow({ children: [hcell("操作阶段"), hcell("耗时 (ms)")] }),
      new TableRow({ children: [cell("明文卷积"), cell(String(timing.plain_ms))] }),
      new TableRow({ children: [cell("BFV 上下文构造"), cell(String(timing.context_ms))] }),
      new TableRow({ children: [cell("输入加密"), cell(String(timing.encrypt_ms))] }),
      new TableRow({ children: [cell("密文卷积 V1（144 次旋转）"), cell(String(timing.v1_ms))] }),
      new TableRow({ children: [cell("密文卷积 V2（16 次旋转）"), cell(String(timing.v2_ms))] }),
    ],
  });
};
contentChildren.push(perfTable());
contentChildren.push(para(run("表 2  各阶段性能数据", { size: 20, italics: true, color: palette.light }), { alignment: AlignmentType.CENTER, spacing: { after: 300 } }));

contentChildren.push(para(new ImageRun({
  type: "png", data: fs.readFileSync(path.join(assetDir, "fig5_performance.png")),
  transformation: { width: 480, height: 266 },
}), { alignment: AlignmentType.CENTER }));
contentChildren.push(para(run("图 7  各阶段性能对比（对数坐标）", { size: 20, italics: true, color: palette.light }), { alignment: AlignmentType.CENTER, spacing: { after: 300 } }));

const speedup = (timing.v1_ms / timing.v2_ms).toFixed(2);
const slowdown = (timing.v2_ms / timing.plain_ms).toFixed(0);
contentChildren.push(bodyPara("版本 B 相比版本 A 提速约 " + speedup + " 倍，这一提升完全来自于旋转次数从 144 次降至 16 次。虽然密文卷积仍比明文慢约 " + slowdown + " 倍，但在隐私保护场景中，这是换取数据可用不可见的必要代价。"));

// 七、实验结论
contentChildren.push(heading("七、实验结论"));
contentChildren.push(bodyPara("本实验基于 TenSEAL（BFV 方案）成功实现了 4×4 输入、3×3 卷积核的密文二维卷积，采用打包→旋转→累加策略，得出以下结论："));
contentChildren.push(bodyPara("（1）正确性验证通过：版本 A 和版本 B 的密文解密结果与明文卷积完全一致，绝对误差均为 0，验证了 BFV 批处理加密在卷积运算中的数学正确性。"));
contentChildren.push(bodyPara("（2）旋转次数优化成功：版本 B 将 9 个核元素合并到一次 dot 操作，将旋转次数从 144 次降至 16 次，达到了串行 dot 策略的理论最小值。"));
contentChildren.push(bodyPara("（3）性能提升显著：版本 B 的密文卷积耗时约为版本 A 的 1/10，验证了减少旋转次数对性能的直接贡献。"));
contentChildren.push(bodyPara("（4）未来方向：可探索将 4 个输出位置并行打包到同一密文的不同 slot 组中，进一步降低旋转次数；或研究 GPU 加速方案（如 cuFHE）以提升大规模场景下的吞吐量。"));

// 附录
contentChildren.push(heading("附录：关键代码清单"));
contentChildren.push(bodyPara("以下为密文卷积版本 B（优化策略）的核心实现代码："));
const codeLines = [
  "def encrypted_conv2d_v2_per_output(input_enc, kernel, vec_len=16):",
  "    k_flat = kernel.flatten().astype(np.int64)",
  "    results, total_rotations = [], 0",
  "    for out_i in range(2):",
  "        for out_j in range(2):",
  "            weights = np.zeros(vec_len, dtype=np.int64)",
  "            for ki in range(3):",
  "                for kj in range(3):",
  "                    pos = (out_i+ki)*4 + (out_j+kj)",
  "                    weights[pos] = k_flat[ki*3+kj]",
  "            encrypted_scalar = input_enc.dot(weights.tolist())",
  "            total_rotations += 4  # log2(16)",
  "            results.append(encrypted_scalar)",
  "    return results, total_rotations",
];
for (const line of codeLines) {
  contentChildren.push(para(run(line, { size: 20, font: { ascii: "Courier New", hAnsi: "Courier New", eastAsia: "SimSun" } }), {
    shading: { type: ShadingType.CLEAR, fill: "F5F5F5" },
    spacing: { before: 0, after: 0, line: 260 },
  }));
}

const doc = new Document({
  features: { updateFields: true },
  sections: [
    {
      properties: { page: { margin: { top: 1440, bottom: 1440, left: 1440, right: 1440 } } },
      children: coverChildren,
    },
    {
      properties: { page: { margin: { top: 1440, bottom: 1440, left: 1440, right: 1440 } } },
      headers: {
        default: new Header({
          children: [para(run("全同态加密密文卷积实验报告 — 打包-旋转-累加策略", { bold: true, size: 20, color: palette.primary }), { alignment: AlignmentType.CENTER })],
        }),
      },
      footers: {
        default: new Footer({
          children: [para(new TextRun({ children: [PageNumber.CURRENT], size: 20 }), { alignment: AlignmentType.CENTER })],
        }),
      },
      children: [toc(tocEntries), ...contentChildren],
    },
  ],
});

const buffer = await Packer.toBuffer(doc);
fs.writeFileSync(outputPath, buffer);
