# OOP

中国科学院大学《面向对象程序设计》课程作业仓库。

## Assignment 0 — 调研报告：服务化后的软件发展新特征

调研软件形态从**独立化 → 产品化 → 结构化 → 对象化 → 网络化 → 服务化**的演化历程，分析总结"服务化"之后软件发展的新特征（如 SOA / SaaS / PaaS、云计算与 API 经济等）。以 Word 形式提交，篇幅 1000–10000 字，列出重要参考文献。

交付文件：`Assignment0/调研报告_服务化后的软件发展新特征.docx`

## Assignment 1 — UML 类图设计

学习用 UML 类图表示"类"，完成两道设计题：

1. **图书馆借书、还书场景**：设计 `Student`（学生）、`Book`（书籍）、`Librarian`（管理员）三个类，定义属性与方法，并表达借阅关联（1 对 0..*）与封装职责。
2. **三视角下的"汽车"**：分别从厂家（产品）、车主（交通工具）、交通管理机构（监管对象）三个角度设计 3 个"汽车"类，体现"视角决定抽象边界"。

交付文件（`Assignment1/`）：

- `题目1-图书馆借还书类图.png` / `题目2-三视角汽车类图.png` — 类图（PlantUML 导出）
- `assignment1_library.puml` / `assignment2_car.puml` — 类图源文件
- `类设计文档.md` — 详细类设计（属性、方法、关系与分析）

本地重新渲染类图：

```powershell
C:\Users\NO.9\.local\bin\jre21\jdk-21.0.12.1+1-jre\bin\java.exe -jar C:\Users\NO.9\.local\bin\plantuml\plantuml.jar -tpng -charset UTF-8 .\Assignment1\assignment1_library.puml
```
