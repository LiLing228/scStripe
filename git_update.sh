#!/bin/bash
# 自动提交并推送代码修改到 GitHub（自动生成 commit message）

# 自动生成 commit message
msg="Update: $(date '+%Y-%m-%d %H:%M:%S')"

# 添加所有修改
git add .

# 提交
git commit -m "$msg"

# 同步远端，避免冲突
git pull origin main --rebase

# 推送
git push origin main