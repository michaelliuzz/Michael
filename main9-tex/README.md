# main9.pdf → LaTeX 转换结果

## 目录结构

```
main.tex              主文档（正文、公式、表格、图、参考文献、附录）
preamble.tex          导言区（字体、版式、章节编号、代码清单样式）
figures/fig1..12.png  从原 PDF 无损提取的 12 幅插图
code/robot_q3_v2.py   附录 B.1 源码（438 行，已验证可被 Python 解析）
code/robot_q4.py      附录 B.2 源码（519 行，已验证可被 Python 解析）
main.pdf              本地 XeLaTeX 编译产物（41 页）
```

## 编译

```bash
xelatex main.tex
xelatex main.tex      # 第二遍用于解析图表交叉引用
```

必须用 **XeLaTeX**（原文即 XeTeX 输出，中文依赖 ctex + Fandol 字体）。

## 说明

1. **章节编号**：正文用「一、二、…」，小节用 `5.1` / `5.1.1`，附录用 `附录 A` / `B.1`，
   与原文一致，由 `preamble.tex` 中的 `\ctexset` 控制。
2. **字体**：优先 `newtxtext + newtxmath`（Times 正文 + Times 数学，完整 TeX Live /
   Overleaf 自带）；无 newtx 时退回 TeX Gyre Termes；两者都无时用 LaTeX 默认字体，
   文档仍可编译。中文用 ctex 默认的 Fandol 字族，与原 PDF 相同。
3. **公式**：13 个编号公式 (1)–(13) 全部按原文重排为 `equation` / `align` 环境，
   编号与原文一一对应；正文中的「式 (2)」「式 (10)」等引用保持原有编号。
4. **源码附录**：原 PDF 中过长的代码行被页边距**截断**（最长仅 147 字符，语法不完整）。
   本转换从 PDF 内容流中恢复了被截断的完整代码，并改用 `listings` 的 `breaklines`
   自动折行显示，因此附录代码比原 PDF 更完整、可直接复制运行。
5. **页数**：41 页（原文 39 页），差异来自代码折行与图表浮动位置，正文内容完全一致。
