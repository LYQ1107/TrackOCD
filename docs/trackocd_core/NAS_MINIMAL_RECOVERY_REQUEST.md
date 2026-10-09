# NAS source-only GitHub recovery request

Paste the following into NAS thread **盘点当前文件夹**. The user selected
**NAS → isolated GitHub recovery branch → A100**, avoiding a client-mediated
archive transfer when NAS authentication is available. This is source-only
delivery, not permission to train, restart old processes, read TAO Test data,
or migrate historical runtime caches. A100's working GitHub proxy/credentials
do not establish NAS authentication; verify it at the source first.

```text
接续刚完成的 TrackOCD v2 核验，按用户选择改为仅将源码推送 GitHub。
A100 已恢复官方 DINOv2 权重且哈希匹配；不复制数据集、模型或 features。

源根：/home/Valadmin/A100/data1/LWR/vranlee/SERVER_ONLY/avis/OCD_OVMOT
运行证据根：/home/Valadmin/A100/usr_for_deadline/trackocd_v2/project_outputs

1. 先读适用 AGENTS.md；若源根有 Git，报告 branch/HEAD/status 和 v2 差异，
   不 checkout/reset/stash，不覆盖或提交源树。若无可用 Git，明确记录。
2. 先列出以下精确范围的普通文件、软链接、字节数：src/trackocd_v2、
   scripts/trackocd_v2、configs/trackocd_v2、tests/trackocd_v2、
   docs/trackocd_v2、research_log.md。源码可以包含历史 Test 入口，
   但绝不执行它们，也不读取任何 Test 数据/标注/实验输出。
3. 从运行证据根只选与当前 frontend/representation/formal-cache 有关的
   小 JSON 配置、状态和 provenance；可包含 cache_manifest.json。
   不选择 features 数组/GT JSON、模型、图片、原始 annotations 或大 JSONL。
   缺失文件记录为缺失；拒绝沿失效软链接猜测映射。
4. 先验证 https://github.com/LYQ1107/TrackOCD 的访问/认证。不要复制私钥、
   打印凭据、改共享 proxy/SSH 或把 A100 认证可用当成 NAS 已认证。
   若上述 source-only 内容总计超过 10 MiB，先报告清单等待缩小范围。
5. 在唯一任务临时目录中创建隔离克隆，以远端 codex/trackocd-v2 的
   cac66467af7ca137fc53bdbef34896dde47030d8 为基底，新建
   codex/trackocd-v2-nas-recovery。若分支已存在，不覆盖；报告并使用唯一
   新名称。原 NAS 工作树始终只读，不 checkout/reset/stash/commit 源树。
6. 仅复制并提交上面已盘点的真实源码/配置/测试/必要说明和小型 provenance。
   对忽略的 docs/元数据只允许明确具名添加，不 git add -f 整个 outputs。
   排除 .git、__pycache__、venv、数据、权重、features 及软链接目标；
   不上传 Test 数据或相关实验输出。附普通文件 SHA256/大小清单，
   区分项目源与运行元数据，缺失项如实记录；不从聊天重新生成代码。
7. 做静态/不读数据的测试后，普通 commit + push 新恢复分支，禁止 force push，
   不覆盖 main、原 codex/trackocd-v2 或当前 TrackOCD core 研究分支。
   输出源 Git HEAD/status、恢复 commit SHA、文件清单/哈希和 ls-remote
   独立核验结果。不要创建 PR、启动服务、训练或恢复旧 supervisor。
   如果访问/认证失败，仅报告具体缺口；源码小包作为后备路线，暂不大迁移。
```

The limit above covers selected source/metadata, not the small isolated Git
clone, and does not change the research goal or its 15/30-GiB budgets. After
independent remote verification, A100 fetches only the returned recovery
branch, reads its source inventory/instructions/configs and compares hashes
against NAS evidence and the preserved local worktree. A100 must not blanket
merge a legacy supervisor or run Test-capable entry points. No claim of complete
v2 recovery may be made from a file inventory or unverified push alone.
