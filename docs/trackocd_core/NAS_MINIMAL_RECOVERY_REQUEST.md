# NAS source-only recovery request

Paste the following into NAS thread **盘点当前文件夹**. This is a minimal
source export request, not permission to train, restart old processes, read
TAO Test data, or migrate historical runtime caches.

```text
接续刚完成的 TrackOCD v2 只读核验。A100 已恢复官方 DINOv2 权重且哈希匹配；
下一步只恢复真实旧 v2 源码和小型配置/证据，不复制数据集、模型或 features。

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
4. 若上述 source-only 范围总计不超过 10 MiB，在新建、唯一的任务临时目录
   创建一个代码恢复包及普通文件 SHA256/大小清单。清单用相对路径，区分
   项目源与运行元数据，保留完整真实源码，不用聊天重写。排除 .git、
   __pycache__、venv、数据、权重及软链接目标。不修改任何原始文件。
   如果超过 10 MiB，先报告清单并等待缩小范围，不启动大迁移。
5. 输出恢复包的绝对路径、字节数、SHA256、源文件清单及 Git 证据。
   不启动 HTTP 服务/后台进程，不改 SSH/proxy，不 push Git，不训练，
   不恢复 supervisor，不执行任何 Test 入口。若没有 A100 传输通道，
   只报告准备好的小包，后续由客户端 scp 中转。
```

The limit above is a first-pass transfer cap, not a change to the research goal
or its 15/30-GiB budgets. After actual transfer, A100 must verify the bundle,
extract into an isolated task recovery directory, read instructions/configs,
and compare exact source bytes against the preserved working tree. No new
claim of complete v2 recovery may be made from a file inventory alone.
