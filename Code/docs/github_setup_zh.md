# 在 GitHub 创建和上传项目

建议仓库名：`sem-grain-segmentation`。
英文简介可填：`U-Net++ with encoder SE attention for SEM grain segmentation and morphology-based instance separation.`

本地项目已完成整理，尚未创建远程仓库。优先使用打包好的
`sem-grain-segmentation-github.zip`，其中只有源码和文档，没有 .venv、本地输出、
合成测试数据或权重。

## 网页创建与上传

1. 登录 GitHub，点击右上角 “+” → “New repository”，或打开
   [创建仓库页面](https://github.com/new)。
2. Repository name 填 `sem-grain-segmentation`，Description 使用上面的简介。
   需要在稿件中写 publicly available 时，仓库必须实际可公开访问。
   尚在整理时可先设 Private，提交论文前再公开。
3. 本地已含 README 和 .gitignore，因此新建页面无需再次初始化这些文件。
   软件许可由作者确定后添加；不要把随意选的许可证当作编辑要求。
4. 点击 “Create repository”。在空仓库点击 “uploading an existing file”；
   已有仓库可选 “Add file” → “Upload files”。
5. 解压源码 ZIP，打开其中的 sem-grain-segmentation 文件夹，把里面的文件与子文件夹
   拖到上传区域，保持目录结构，使 README.md 位于仓库根目录。
   不要直接把 ZIP 文件作为唯一内容上传。
6. 在提交说明中填写 `Add SEM segmentation code and documentation`，
   完成页面上的提交操作；如果选择了新分支，按页面提示创建并合并 PR。
7. 查看 README 是否正常显示，检查 run.py、configs、semseg 等目录是否完整。
   用未登录窗口确认公开仓库能访问。
8. 复制真实仓库地址，替换论文中的 [GITHUB_REPOSITORY_URL]。

来源：[GitHub 官方文件上传说明](https://docs.github.com/en/repositories/working-with-files/managing-files/adding-a-file-to-a-repository)。

网页上传不会按本机 .gitignore 自动替你排除文件。因此建议上传源码 ZIP 的解压内容，
不要把含 .venv 或 outputs 的整个开发目录拖入网页。ZIP 中 metadata 文件只有表头，
不是虚构的实际划分表。

## 也可使用 Git 命令

在 GitHub 先创建空仓库，不初始化 README、.gitignore 或 LICENSE。
以下命令供作者自行执行，YOUR_USERNAME 要替换成真实用户名。

```powershell
Set-Location -LiteralPath 'C:\Users\work\Documents\DJF\sem-grain-segmentation'
git init -b codex/sem-grain-release
git add README.md run.py requirements.txt requirements-tested-cpu.txt .gitignore configs semseg scripts tests legacy docs data/README.md metadata/split_manifest.template.csv
git diff --cached --stat
git commit -m "Add revised SEM segmentation workflow"
git remote add origin https://github.com/YOUR_USERNAME/sem-grain-segmentation.git
git push -u origin codex/sem-grain-release
```

严格在项目目录内执行，不能把上一级含其他个人文件的 DJF 目录作为上传对象。
按 Git 提示完成真实姓名/邮箱设置与 GitHub 登录；本指南没有替作者创建账号、提交
代码或推送远程仓库。分支名称可以按团队约定调整。

来源：[GitHub 官方本地代码上传说明](https://docs.github.com/en/migrations/importing-source-code/using-the-command-line-to-import-source-code/adding-locally-hosted-code-to-github)。

## 最终发布版本

在 README 和 data/README.md 填实际论文题目、作者及通讯作者联系渠道；
确认软件使用许可并添加 LICENSE。用真实薄膜记录完成训练与独立测试后，核对
论文结果与代码，固定论文对应的 commit，再创建实际 tag/release。

如需 DOI，可按
[GitHub 科研代码归档说明](https://docs.github.com/en/repositories/archiving-a-github-repository/referencing-and-citing-content)
将公开版本归档到 Zenodo。得到真实 DOI 后再填入论文，不提前编造。

大型权重不包含在本次源码包中。GitHub 网页上传单文件不超过 25 MiB，
普通 Git 仓库阻止大于 100 MiB 的文件；若决定公开权重，可考虑 Git LFS 或
release 附件，并核对对应限制。
来源：[GitHub 大文件说明](https://docs.github.com/en/repositories/working-with-files/managing-large-files/about-large-files-on-github)。

## 作者完成真实实验的运行顺序

1. 使用 Python 3.10/3.11 安装依赖；README 有 CPU/GPU 环境说明。
2. 把真实图像和 mask 按 data/README.md 放好，填写 metadata/split_manifest.csv。
3. 运行 scripts/validate_split.py 检查划分。
4. 运行 run.py train 完成 50 epochs 训练并按验证 Dice 保存最佳模型。
5. 固定模型和参数，运行 run.py evaluate --split test 输出逐图/逐薄膜指标。
6. 核对结果后更新正文、SI 与 Response 1，再填写真实 GitHub 链接。

源码包中的合成测试只证明程序流程可执行，不代表正式 SEM 实验已经完成。
