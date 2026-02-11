# 快速部署说明

## 🚨 如果遇到包安装错误

如果运行 `bash raspberry_pi_setup.sh` 时遇到：
- `Unable to locate package libjasper-dev`
- `Package has no installation candidate`

**请使用虚拟环境版本：**

```bash
bash raspberry_pi_setup_venv.sh
```

然后使用：

```bash
bash start_venv.sh
```

---

## 两种部署方式对比

### 方式 1：传统部署（可能失败）
```bash
bash raspberry_pi_setup.sh
bash start.sh
```
适用于：Debian bookworm (stable)

### 方式 2：虚拟环境部署（推荐）⭐
```bash
bash raspberry_pi_setup_venv.sh
bash start_venv.sh
```
适用于：任何 Linux 发行版，包括 Debian trixie/testing

---

## 虚拟环境的优势

- ✅ 不依赖系统包
- ✅ 不破坏系统 Python
- ✅ 完全隔离
- ✅ 兼容性更好

---

查看详细说明：`cat VENV_DEPLOYMENT.md`
