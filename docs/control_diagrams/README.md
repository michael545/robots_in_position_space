# Control Diagrams & Flowcharts

This directory contains the visual diagrams, vector graphics, and editable source definitions for the **Ubiquity Robotics Magni MCB G6** control pipeline.

---

## 1. Primary Control Path: ROS 2 to Motor Physics

### High-Resolution Rendered Pipeline
![Control Path Pipeline](control_path.png)

* **Scalable Vector Graphic:** [control_path.svg](control_path.svg)
* **High-Resolution PNG:** [control_path.png](control_path.png)
* **Graphviz Source File:** [control_path.dot](control_path.dot)
* **Mermaid Source File:** [control_path.mmd](control_path.mmd)

---

## 2. How to View and Edit Online

### A. Editing via Mermaid Live Editor
You can copy the contents of [`control_path.mmd`](control_path.mmd) directly into [Mermaid Live Editor](https://mermaid.live).
1. Open [https://mermaid.live](https://mermaid.live).
2. Paste the text from [`control_path.mmd`](control_path.mmd) into the left code editor.
3. Edit nodes, labels, or equations in real time with instant graphical preview.
4. Export as SVG, PNG, or Markdown.

### B. Regenerating Graphviz Diagrams Locally
To recompile the `.dot` file into vector SVG or high-resolution PNG:
```bash
# Render to vector SVG
dot -Tsvg control_path.dot -o control_path.svg

# Render to high-resolution PNG (160 DPI)
dot -Tpng -Gdpi=160 control_path.dot -o control_path.png
```

---

## 3. Directory Contents

| File | Format | Purpose |
| :--- | :--- | :--- |
| [`control_path.svg`](control_path.svg) | SVG (Vector) | Lossless scalable vector diagram for web, presentations, and interactive inspection. |
| [`control_path.png`](control_path.png) | PNG (Raster) | High-resolution raster rendering for markdown previewers, IDEs, and documentation. |
| [`control_path.dot`](control_path.dot) | Graphviz DOT | Primary compiled source diagram with ortholinear graph styling and color-coded subgraphs. |
| [`control_path.mmd`](control_path.mmd) | Mermaid | Pure markdown flowchart for GitHub preview and online editing at [mermaid.live](https://mermaid.live). |
