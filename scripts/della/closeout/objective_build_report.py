"""Build the objective hill-climb close-out HTML report from the arms json (numbers) + the prose below.

    python3 objective_build_report.py <arms.json> <out.html> <commit> "<push status html>"
"""
import html, json, sys
A = json.load(open(sys.argv[1]))
OUT, COMMIT, PUSH = sys.argv[2], sys.argv[3], sys.argv[4]
arms = {r["label"]: r for r in A["arms"]}
WB = "https://wandb.ai/reself/marin-della/runs/"
e = html.escape

def f(x, nd=4, sign=True):
    return (f"{x:+.{nd}f}" if sign else f"{x:.{nd}f}").replace("-", "−")

def tp(s):
    return f"t {f(s['t'], 2)}, p {s['p']:.3f}"

def cell(s, nd=4):
    return f"{f(s['d'], nd)} <span class=dim>({f(s['t'], 2)})</span>"

def sig(s, bonf=3.97):
    return "good" if s["t"] < -bonf else ("warn" if s["t"] > bonf else "")

def link(rid, text=None):
    return f'<a href="{WB}{e(rid)}">{e(text or rid)}</a>'

pm = A["pool130"]["macro"]; pm3 = A["pool300"]["macro"]
import statistics as st
pool_line = f"130m 池 n={len(pm)}：macro 均值 {st.mean(pm):.5f}、sd {st.stdev(pm):.5f}；300m 池 n={len(pm3)}：{st.mean(pm3):.5f}、sd {st.stdev(pm3):.5f}"

fam_rows = [
    ("twin", "反向模型状态匹配（Twin Networks）", "twin w0.1 (free head)", "c4_en bpb +0.0013（t 3.7）", "关闭：净成本，门控前后都无收益"),
    ("sr", "后继表示 TD（successor representation）", "sr w0.1 gated 4.0:3.6", "无域过 Bonferroni", "关闭：门控下只是「免费」而非「有用」；不门控 w0.1 为 +0.011"),
    ("pi", "预测信息 InfoNCE（h<sub>t</sub> 识别 h<sub>t+4</sub>）", "pi k4 w0.1", "c4_en bpb +0.0053（t 15）", "关闭：主动伤害；读 L3 可退掉大半代价但无收益"),
    ("eos", "到文档末的距离（13 档分类）", "eos w0.1", "c4_en bpb +0.0036（t 10）", "关闭：门控把 c4_en 代价退掉约八成，仍无收益"),
    ("mtp", "多 token 预测（D×D 投影 + 共享 lm_head）", "mtp k2 w0.1 gated", "c4_en bpb +0.0015（t 4.1）", "关闭：门控也退不掉它的早期损伤"),
    ("dn", "在自身样本腐蚀的输入上做去噪 NTP", "dn w0.1 gated", "redpajama −0.021（t −3.8）、code −0.042", "关闭：结构化文本的收益被社交文本域的代价抵平，macro 为零"),
    ("ebm", "判别真实前缀与模型自身一步样本的 NCE（能量式）", "ebm w0.03 gated (FIXED code)", "code / redpajama / dolma-v1_5 均过 Bonferroni", "关闭：逐域真实，但无 macro 净收益，300m 未达门槛"),
    ("swap", "真实文本、错误上下文（跨文档拼接）的判别", "swap w0.03 gated", "c4_en、wikitext 各差约 2 个池 sd", "关闭：伤了它本该帮的散文域"),
    ("adv", "判别器分数作生成侧 REINFORCE 奖励", "adv w0.03 ungated", "c4_en bpb +0.0034（t 9.6）", "关闭：门控下与同代码 ebm 臂分辨不出差异；不门控给散文加税"),
]
fam_html = []
for fam, sig_desc, lab, dom, verdict in fam_rows:
    r = arms[lab]; m = r["macro"]
    fam_html.append(f"<tr><td><b>{fam}</b></td><td>{sig_desc}</td><td>{e(lab)} <span class=dim>(n={r['n']})</span></td>"
                    f"<td class=num>{f(m['d'],5)}<br><span class=dim>{tp(m)}</span></td><td>{dom}</td><td>{verdict}</td></tr>")

ladder = ["sr w0.1 gated 4.0:3.6", "dn w0.1 gated", "ebm w0.03 ungated", "ebm w0.03 gated (pre-fix code)", "ebm w0.03 gated (FIXED code)"]
ladder_name = {"sr w0.1 gated 4.0:3.6": "门控 + 任意辅助项（sr）", "dn w0.1 gated": "门控 + 腐蚀输入，无判别器（dn，修复前）",
               "ebm w0.03 ungated": "判别器，不门控（修复前）", "ebm w0.03 gated (pre-fix code)": "判别器 + 门控（修复前代码）",
               "ebm w0.03 gated (FIXED code)": "判别器 + 门控（修复后代码）"}
lad_html = []
for lab in ladder:
    r = arms[lab]
    lad_html.append(f"<tr><td>{ladder_name[lab]}</td><td class=num>{r['n']}</td><td class=num>{cell(r['macro'],5)}<br><span class=dim>p {r['macro']['p']:.3f}</span></td>"
                    f"<td class='num {sig(r['code'])}'>{cell(r['code'])}</td><td class='num {sig(r['redpajama'])}'>{cell(r['redpajama'])}</td>"
                    f"<td class='num {sig(r['dolma'])}'>{cell(r['dolma'])}</td><td class='num {sig(r['c4_bpb'])}'>{cell(r['c4_bpb'],5)}</td></tr>")

app_html = []
for r in A["arms"]:
    m = r["macro"]
    rid = r["run_ids"][0]
    app_html.append(f"<tr><td>{r['size']}</td><td>{link(rid, r['label'])}</td><td class=num>{r['n']}</td>"
                    f"<td class=num>{f(m['d'],5)}</td><td class=num>{f(m['t'],2)}</td><td class=num>{m['p']:.3f}</td>"
                    f"<td class=num>[{f(m['lo'],4)}, {f(m['hi'],4)}]</td><td class='num {sig(r['code'])}'>{cell(r['code'])}</td>"
                    f"<td class='num {sig(r['redpajama'])}'>{cell(r['redpajama'])}</td><td class='num {sig(r['c4_bpb'])}'>{cell(r['c4_bpb'],5)}</td></tr>")

def runs_of(lab):
    return " · ".join(link(x, x.split("della4xh100")[-1] or "(seed 0)") for x in arms[lab]["run_ids"])

pool_ids = ["muonh-qwen3-130m-della4xh100" + s for s in ["", "-s1", "-s2", "-s3", "-ema0.999", "-ema0.999-s1", "-ema0.999-s2", "-ema0.999-s3"]]
pool3_ids = ["muonh-qwen3-300m-della4xh100" + s for s in ["", "-s1", "-s2", "-s3"]]
fx, pre, e300 = arms["ebm w0.03 gated (FIXED code)"], arms["ebm w0.03 gated (pre-fix code)"], arms["ebm w0.03 gated 3.65:3.23 (300m escalation)"]

doc = f"""<!doctype html>
<html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Objective Hill-Climb 结档报告</title>
<style>
:root{{--bg:#fbfaf7;--fg:#1d1d1f;--dim:#6b6b70;--line:#e3e1db;--card:#fff;--acc:#1f5fbf;--good:#1e7a3c;--goodbg:#e8f4ec;--warn:#a1331f;--warnbg:#f8e9e5;--lead:#f1f5fb}}
@media (prefers-color-scheme:dark){{:root{{--bg:#16171a;--fg:#e8e6e1;--dim:#9a9aa2;--line:#2e3036;--card:#1d1f23;--acc:#7fb0ff;--good:#6fd08f;--goodbg:#173323;--warn:#ff9c85;--warnbg:#3a1f1a;--lead:#1a2433}}}}
*{{box-sizing:border-box}} body{{margin:0;background:var(--bg);color:var(--fg);font:15px/1.65 -apple-system,BlinkMacSystemFont,"Segoe UI","PingFang SC","Hiragino Sans GB","Noto Sans CJK SC","Microsoft YaHei",sans-serif}}
main{{max-width:1080px;margin:0 auto;padding-block:32px 64px;padding-inline:20px}}
h1{{font-size:26px;margin:0 0 6px}} h2{{font-size:19px;margin:36px 0 10px;padding-top:6px;border-top:1px solid var(--line)}} h3{{font-size:16px;margin:22px 0 6px}}
.meta{{color:var(--dim);font-size:13px}} .dim{{color:var(--dim);font-size:12px}} a{{color:var(--acc);text-decoration:none}} a:hover{{text-decoration:underline}}
.lead{{background:var(--lead);border-left:4px solid var(--acc);padding:14px 18px;border-radius:6px;margin:18px 0;font-size:16px}}
.badge{{display:inline-block;font-size:12px;padding:1px 8px;border-radius:10px;background:var(--warnbg);color:var(--warn);margin-left:8px;vertical-align:2px}}
.wrap{{overflow-x:auto;margin:10px 0;border:1px solid var(--line);border-radius:6px;background:var(--card)}}
table{{border-collapse:collapse;width:100%;font-size:13.5px}} th,td{{padding:7px 10px;border-bottom:1px solid var(--line);text-align:left;vertical-align:top}}
th{{font-weight:600;background:var(--bg);white-space:nowrap}} tr:last-child td{{border-bottom:none}} td.num{{text-align:right;font-variant-numeric:tabular-nums;white-space:nowrap}}
td.good{{background:var(--goodbg);color:var(--good)}} td.warn{{background:var(--warnbg);color:var(--warn)}}
code,pre{{font-family:ui-monospace,SFMono-Regular,Menlo,Consolas,monospace;font-size:12.5px}} pre{{background:var(--card);border:1px solid var(--line);border-radius:6px;padding:10px 12px;overflow-x:auto;white-space:pre}}
ul{{padding-left:20px}} li{{margin:3px 0}} .small{{font-size:13px}}
</style></head><body><main>
<h1>LLM 目标函数爬坡（objective hill-climb）结档报告 <span class=badge>已结档 2026-09-22</span></h1>
<div class=meta>2026-09-18 → 2026-09-22 · 仓库 <code>marin</code> 分支 <code>objective-hillclimb</code> @ <code>{e(COMMIT)}</code> · W&amp;B <a href="https://wandb.ai/reself/marin-della">reself/marin-della</a> · 研究日志 <code>docs/della/objective-hillclimb.md</code></div>

<div class=lead><b>一句话结论：</b>在数据、步数、batch、架构与优化器全不变的 Kaiyue Wen <code>muonh_qwen3</code> baseline 上，9 个辅助目标族（70 条完整 run）没有一个显著降低 Paloma macro。唯一真实的正效应是门控早期能量式 NCE（判别模型自身的一步样本），它只可靠改善 code、redpajama、dolma-v1_5 三个结构化文本域；修复文档边界 bug 后，它的 macro 为 {f(fx['macro']['d'],4)}（p {fx['macro']['p']:.2f}）；300m 升档（修复前代码）也未达预注册门槛。</div>

<h2>1. 任务、约束与判据</h2>
<ul>
<li><b>目标</b>（用户，2026-09-18）：找一个比 next-token prediction（或 NTP + 辅助项）更 data-efficient 的训练目标。<b>data efficiency</b> 按用户定义：每条臂都把同一份定量数据跑完，compute 不计，比终值；不把收益换算成「NTP 需要多少额外数据」。</li>
<li><b>不变量</b>：数据（fineweb-edu-10B 预分词，data_seed 42）、步数与 schedule、batch、架构、MuonH 优化器都不改。eval 永远是纯 NTP 前向，辅助项只在训练时存在。辅助头用 <code>FreeLinear</code> 放进 adam 组，因为 MuonH 会把任何 <code>hnn.Linear</code> 钉在初始范数。</li>
<li><b>规模</b>：130m（hidden 512 · 6 层 · batch 128×4096 · 4959 步 ≈ 2.6B token）；300m（hidden 768 · 12 层 · 11444 步 ≈ 6.0B token）。4×H100 / Della。</li>
<li><b>判据</b>：Paloma macro，即 16 个域 token 加权损失的等权平均；用户在 09-20 定下此判据，取代只看 c4_en。所有比较一律经 <code>scripts/della/objective_compare.py</code>：只收 finished 且到达最终步的 run；两臂须同预算；16 个域在两侧都必须齐全；报 Welch df / p / t 区间；每次 19 个检验，做 Bonferroni。{pool_line}。</li>
<li><b>否决</b>（用户）：自蒸馏；任何「NTP 变体」，即标签平滑、token 重加权、替代 scoring rule 这类改写 NTP 目标的做法。</li>
</ul>

<h2>2. 候选族一览（macro Δ 均为对 baseline 池，负 = 更好）</h2>
<div class=wrap><table><thead><tr><th>族</th><th>学习信号</th><th>代表臂</th><th>macro Δ</th><th>关键逐域</th><th>结局与原因</th></tr></thead><tbody>
{''.join(fam_html)}
<tr><td><b>ebm_steps</b></td><td>k 步自回归自身样本负样本（ebm 的延伸）</td><td>k=4, n=4</td><td class=num>—</td><td>—</td><td>已实现，CPU smoke 通过；n=4 作业在结档时于排队中取消，<b>从未运行</b>（见 §6）</td></tr>
</tbody></table></div>
<p class=small><b>代码版本</b>：用到自身样本腐蚀的族是 dn、ebm、adv，文档边界 bug 只影响它们。dn、ebm 的早期各臂与 300m 升档跑的是修复前代码；ebm「FIXED code」一行与 adv 跑的是修复后代码。其余族不涉及这段代码。代表臂取该族信息量最大的一条（n≥4 优先）。全部 32 个臂（130m 31 个、300m 1 个）的数字见附录。n=1 的臂用 one-vs-sample se（sd·√(1+1/8)，df 7），只作筛选，不作结论。</p>

<h2>3. 主线：ebm 的机制与结局</h2>
<h3>3.1 机制阶梯（n≥4，对 8 条 baseline 池；括号内为 t；绿 = 在有利方向过 Bonferroni |t|&gt;3.97，红 = 在不利方向过）</h3>
<div class=wrap><table><thead><tr><th>臂</th><th>n</th><th>macro Δ</th><th>code Δ</th><th>redpajama Δ</th><th>dolma-v1_5 Δ</th><th>c4_en bpb Δ</th></tr></thead><tbody>
{''.join(lad_html)}
</tbody></table></div>
<p>读法：结构化文本上的收益来自判别器。门控本身、以及「腐蚀输入但不判别」都拿不到多少。门控的作用是去掉代价：不门控时 c4_en bpb {f(arms['ebm w0.03 ungated']['c4_bpb']['d'],5)}（t {f(arms['ebm w0.03 ungated']['c4_bpb']['t'],2)}），门控后接近零，而结构化文本的收益不变。所以两个成分缺一不可。</p>

<h3>3.2 旋钮都已关闭（均为修复前代码）</h3>
<p>权重 0.015 / 0.03 / 0.05 / 0.10：code 上是倒 U，macro 在 0.03 处最好，工作点定在 0.03。0.015 的 n=1 筛曾领先，到 n=4 时 macro 只剩 {f(arms['ebm w0.015 gated']['macro']['d'],4)}（p {arms['ebm w0.015 gated']['macro']['p']:.2f}）。采样温度 2（n=4）：macro {f(arms['ebm w0.03 T2 gated']['macro']['d'],4)}（p {arms['ebm w0.03 T2 gated']['macro']['p']:.2f}），它的 n=1 筛也曾领先。腐蚀率 ρ 0.8 / 1.0 都未过预注册闸门（code ≤ −0.10 且 macro 低于池 2 se）。读出层 L3 丢掉了 code 收益。三种门控关门时点在 n=1 上分辨不出。</p>

<h3>3.3 文档边界 bug 及其量化</h3>
<p>ebm 的腐蚀样本 <code>samp[t]</code> 取自 h<sub>t−1</sub>。在打包窗口里，文档起始位置 t 的 h<sub>t−1</sub> 属于上一篇文档，而旧代码只保护了 position 0。于是被替换的 token 是上下文错位的拼接，判别器不必学什么就能识破；而且 <code>informative</code> 掩码会把该文档的整个后缀都算作被评分的负样本。</p>
<p>它的规模：从分词缓存实测平均文档长 1003.3 token，即每个 4096 窗口 4.08 篇；再结合 E[ρ]=0.25，<b>推算</b>约 19% 的被评分位置受影响。</p>
<p>修复（每个文档起始都受保护）后跑 n=4 对照。按事先冻结的 <code>objective_fraction.py</code> 算，修复抹掉的 code 收益份额 f = 0.296（bootstrap 95% [0.088, 0.467]，delta 法 [−0.027, 0.619]），redpajama f = 0.267。两个区间对「是否过半」结论不一致，按冻结规则只报不判。</p>
<p>修复后，code / redpajama / dolma-v1_5 仍过 Bonferroni；macro 从 {f(pre['macro']['d'],5)}（p {pre['macro']['p']:.3f}）降到 {f(fx['macro']['d'],5)}（p {fx['macro']['p']:.3f}），不再显著。</p>

<h3>3.4 升到 300m：未达预注册门槛</h3>
<p>预注册于升档前（09-20）：code ≤ −0.05 且 macro 为负。300m 的 8 段续跑链跑的是<b>修复前代码</b>：链运行期间主仓库冻结编辑，修复只在旁边的 worktree 里测试，等链全部结束才合并。所以这里应当对照 130m 修复前的臂。n=4 对 n=4 的结果：code {f(e300['code']['d'],4)}（t {f(e300['code']['t'],2)}），macro {f(e300['macro']['d'],5)}（p {e300['macro']['p']:.2f}），c4_en bpb {f(e300['c4_bpb']['d'],5)}。领先的仍是同三个域、方向也相同，但 code 的幅度只有 130m 修复前臂（{f(pre['code']['d'],4)}）的三分之一左右。</p>
<p>这一判负在第 4 个种子落地前就已写下。此前的 n=1 读数（code −0.056、macro −0.013）是本项目第三次同形状的胜者诅咒：前两次是 T2 和 w=0.015，都是 n=1 时强、n≥4 回归到零。</p>

<h2>4. ebm 之后的两族，以及跨族综合</h2>
<ul>
<li><b>swap</b>（真实文本、错误上下文）：本意是补 ebm 在散文域的空白。筛的结果是 macro {f(arms['swap w0.03 gated']['macro']['d'],5)}，c4_en 与 wikitext 各差约 2 个池 sd——它恰恰伤了本该帮的域。</li>
<li><b>adv</b>（ebm 判别器作生成侧 REINFORCE 奖励，修复后代码）：门控版对 baseline 池 macro {f(arms['adv w0.03 gated']['macro']['d'],5)}，差 0.0017 未过筛选门槛；对<b>同代码</b>的修复后 ebm 臂（n=4）为 −0.0059（t −1.26，df 3，p 0.30），n=1 下分辨不出生成项的作用。不门控版 c4_en bpb {f(arms['adv w0.03 ungated']['c4_bpb']['d'],5)}（t {f(arms['adv w0.03 ungated']['c4_bpb']['t'],2)}），c4、c4_100_domains、falcon 三域在错误方向上过 Bonferroni，且哪个域都没换回来。它的散文税比不门控 ebm（+0.0013）大，但那条 ebm 臂跑的是修复前代码，所以差值不能归到生成项上。</li>
<li><b>综合（一个解释，不是已证明的结论）</b>：只重新打包数据里已有信息的辅助项（twin、sr、pi、eos、mtp、dn、swap）7/7 为零或负。唯一有净逐域收益的 ebm，注入的是 NTP 拿不到的信息，即模型自身的错误分布；它也只在这些错误可被检测的域起效。在单 epoch、约 20 token/参数的预算下，NTP 已经是对数据的最大似然，只起正则作用的辅助项没有用武之地。</li>
</ul>

<h2>5. 方法学上值得带走的</h2>
<ul>
<li><b>n=1 读数不可信</b>：同形状的胜者诅咒出现了三次（T2、w=0.015、300m 升档），规律是 n=1 时强，到 n≥4 回归零。</li>
<li><b>池不足 4 条时不报 Δ/sd</b>：300m 的 c4_en「反转」在池 n=3 时是 +5.3 sd（因为池 sd 只有 0.00008），第 4 条 baseline 落地后变成 +0.8 sd。</li>
<li><b>三个静默的分析错误都已写进代码守卫</b>：在跑 run 的 W&amp;B summary 是中途值，曾把对照表整列翻转；三个 Paloma 域名与假设不同；n=1 的 Welch 未定义。</li>
<li><b>分析先于数据冻结</b>：判据写进 doc、分析写进脚本并提交，然后才读数（<code>objective_fraction.py</code> 在种子落地前提交）。</li>
<li><b>跨代码版本的比较无效</b>：adv 的第一轮读数曾拿修复后的 adv 去比修复前的 ebm 臂，得出「门控下等于 ebm 均值」「不门控多交 2.8 se 的税」。这是在结档审计中核对每个手写数字的来源时发现的，已在研究日志中更正。</li>
<li><b>自我纠错也留档</b>：曾误报「baseline 池混入 EMA run」。实际上 EMA 不改训练，只在独立的 <code>eval/ema/*</code> 下评估，池没有问题；这次误报当场核实后撤回。</li>
</ul>

<h2>6. 复现与续跑路径</h2>
<ul>
<li><b>代码</b>：
  <ul>
  <li><code>experiments/references/objective_qwen3.py</code>：所有族的实现，由 <code>VARIANT</code> 与各 <code>*_W</code> 环境变量切换。</li>
  <li><code>experiments/references/della_muonh_qwen3_scaling.py</code>：启动器；run id 由变体 tag 自动生成。</li>
  <li><code>scripts/della/muonh_qwen3_h100x4.sbatch</code>：训练作业脚本。</li>
  <li><code>scripts/della/objective_cpu_smoke.py</code>：CPU smoke，约 2.5 分钟，覆盖每个族的记账与梯度。</li>
  </ul></li>
<li><b>分析</b>：<code>scripts/della/objective_compare.py</code>（cell 对池，附全部守卫）和 <code>scripts/della/objective_fraction.py</code>（修复抹掉收益的份额）。示例：
<pre>cd scripts/della &amp;&amp; ../../.venv/bin/python objective_compare.py --size=130m \\
  --pool=,-s1,-s2,-s3,-ema0.999,-ema0.999-s1,-ema0.999-s2,-ema0.999-s3 \\
  --cell=-ebmr0.5w0.03-g4-3.6-fh-fixbnd,-ebmr0.5w0.03-g4-3.6-fh-s1-fixbnd,-ebmr0.5w0.03-g4-3.6-fh-s2-fixbnd,-ebmr0.5w0.03-g4-3.6-fh-s3-fixbnd</pre></li>
<li><b>文档</b>：<code>docs/della/objective-hillclimb.md</code>（逐日研究日志：判据、预注册、结果、撤回）。另有本地台账 <code>logs/objective_hillclimb_qwen3_h100x4.yaml</code>，按约定 gitignored，记录作业号、状态与每次读数的原始数字。</li>
<li><b>未运行的第 4 族</b>（<code>ebm_steps=k</code>，commit <code>46f7e49365</code>，判据见研究日志 §5.4）：位置按长度 k 对齐分块，块内第 j 位在「前 j 位已替换」的序列上用一次无梯度前向抽样。k=1 时逐位等于原 ebm。CPU smoke 用贪心子类逐 pass 重建腐蚀序列，10 个 key 全部逐位一致。续跑命令：
<pre>for S in 0 1 2 3; do sbatch --time=03:15:00 --job-name=obj-130m-ebmk4-s$S \\
  --export=ALL,SIZE=130m,VARIANT=ebm,EBM_W=0.03,EBM_RHO=0.5,EBM_STEPS=4,AUX_GATE=4.0:3.6,SEED=$S \\
  scripts/della/muonh_qwen3_h100x4.sbatch; done</pre></li>
<li><b>推送状态</b>：{PUSH}</li>
</ul>

<h2>7. W&amp;B 关键 run</h2>
<ul class=small>
<li>130m baseline 池（n=8）：{' · '.join(link(x, x.split('della4xh100')[-1] or '(seed 0)') for x in pool_ids)}</li>
<li>ebm 门控，修复前（n=8）：{runs_of('ebm w0.03 gated (pre-fix code)')}</li>
<li>ebm 门控，修复后（n=4）：{runs_of('ebm w0.03 gated (FIXED code)')}</li>
<li>ebm 不门控（n=4）：{runs_of('ebm w0.03 ungated')} · sr 门控（n=4）：{runs_of('sr w0.1 gated 4.0:3.6')} · dn 门控（n=4）：{runs_of('dn w0.1 gated')}</li>
<li>300m 池（n=4）：{' · '.join(link(x, x.split('della4xh100')[-1] or '(seed 0)') for x in pool3_ids)} · 300m ebm（n=4）：{runs_of('ebm w0.03 gated 3.65:3.23 (300m escalation)')}</li>
<li>swap：{runs_of('swap w0.03 gated')} · adv：{runs_of('adv w0.03 gated')} · {runs_of('adv w0.03 ungated')}</li>
</ul>

<h2>附录：全部臂（对 baseline 池；括号内为 t）</h2>
<div class=wrap><table><thead><tr><th>规模</th><th>臂（链接为首个种子）</th><th>n</th><th>macro Δ</th><th>t</th><th>p</th><th>95% CI</th><th>code Δ</th><th>redpajama Δ</th><th>c4_en bpb Δ</th></tr></thead><tbody>
{''.join(app_html)}
</tbody></table></div>
<p class=dim>数字由 <code>objective_compare.py</code> 的 <code>collect</code> / <code>welch</code> / <code>pval</code> 在结档时重新计算，与研究日志中各时点的读数一致。n=1 的 t 用 one-vs-sample se。</p>
</main></body></html>
"""
open(OUT, "w").write(doc)
print("wrote", OUT, len(doc), "bytes")
