# Arbiter demo: talk track

The video (`arbiter_demo.mp4`) is silent and 3:49 long. Read the matching section while it plays; it does not need to be in sync. Lines in *italics* say what is on screen. If you have extra time, the explainer and Q&A are at the end.

---

## Before you press play (about 40 s, video paused)

> Search engines usually work in two stages. First a fast method like BM25 pulls about a hundred candidate passages. Then a **reranker** reads the query and each passage together and puts the best answer on top. You build a reranker by taking a pretrained language model and fine-tuning it on queries with their correct passages.
>
> Recently, people have published **decision models**: versions of ModernBERT-large that were already fine-tuned to answer yes/no and multiple-choice questions about a passage. That sounds a lot like judging relevance. So our question is simple: **if you start training a reranker from one of these decision models instead of the plain ModernBERT, do you get a better reranker?**
>
> We tested this fairly, with eight starting points, the same training recipe for all of them, four amounts of training data, and an analysis we froze before seeing any test results. The demo shows the code running and then the results.

---

## The video

### 0:00 – 0:13 · The project
*`ls`, then the top of the config file, which lists the eight starting points.*

> This is the repository. Everything about the study lives in one config file. These are the eight starting points. **B** is plain ModernBERT-large. **D1 to D5** are five public decision models built on it. Then there are two controls. **N** is a checkpoint trained on natural language inference, to check whether *any* extra supervised training helps. **R** is ModernBERT with random noise added, exactly as large as a decision model's change, to check whether simply *moving* the weights helps.

### 0:13 – 0:26 · Tests
*`pytest -q`, 24 tests pass.*

> The code has a test suite. For example, our ranking metrics are checked against the standard `ir_measures` library, so the numbers we report are computed correctly.

### 0:26 – 1:20 · Training, live, with interrupt and resume
*A real training run on the GPU, sped up 3×. The loss drops, Ctrl-C at about 0:41, and the same command resumes from step 40.*

> This is a real training run, ModernBERT on a thousand MS MARCO queries. Each step shows the model a query with one correct passage and seven wrong ones, and the loss is how badly it ranks them. A random guess gives about 2.08, and you can see it going down.
>
> Now I interrupt it. It saves a checkpoint and stops cleanly. Running the same command again resumes from that exact step. This mattered because the full study was 95 training runs on a single GPU.
>
> At the end it scores itself on 500 development queries and prints the result, a mean reciprocal rank of about 0.53. That means the right passage is usually first or second.

### 1:20 – 2:35 · Does the reranker actually work?
*1:20 shows the query and six passages. 1:26 shows the untrained model's ranking, and 2:00 the ranking from the model we just trained.*

> Here's what a reranker does. The query is "what causes ocean tides", with six passages. One of them is a restaurant called Ocean Tides, which is a keyword trap. **Before training**, the model ranks the actual answer last. **After about a minute of training**, the correct explanation is first, and the restaurant and photosynthesis passages are pushed down. That's the skill our study measures, just at scale.

### 2:35 – 2:50 · The real study and the preregistration
*`arbiter status` shows all runs done. `git tag -n` shows the frozen preregistration tag.*

> This is the actual study: 40 learning-rate search runs, 52 main runs and 3 extra checks, all finished. Before running the main experiments, we froze the analysis code in this git tag, dated 23 November. That means we couldn't change how we measure success after seeing the results. The test scores were sealed and read only once, at the end.

### 2:50 – 3:09 · The results
*`arbiter summary` prints the headline table.*

> Each row is the decision models minus plain ModernBERT, with a 95% interval. The first block is our preregistered test, and the honest answer is **inconclusive**: the intervals cross zero.
>
> The reason is that five training runs **collapsed**; they never really learned. If we set those aside, which is an exploratory analysis decided afterwards, we see a small head start of about 0.03 to 0.05 at a thousand queries, gone by sixteen thousand. So the advantage **fades** with more data.
>
> And it isn't really about decisions. The NLI checkpoint gives a bigger boost, and even random noise gives some.

### 3:09 – 3:49 · Figures
*Five figures, 8 seconds each: 3:09, 3:17, 3:25, 3:33, 3:41.*

1. **Learning curves.**
   > All starting points as training data grows. The others start well ahead of plain ModernBERT, then everyone converges by 16,000 queries. The hollow points are the collapsed runs.
2. **Forest plot.**
   > The effect per decision model and per data size. The wide bars are the preregistered test. The narrow ones are with collapsed runs set aside.
3. **Training loss.**
   > This is a collapse. Laya, at the high learning rate it preferred in the search, starts learning, then falls back towards random guessing. At ModernBERT's learning rate it trains normally. So the problem is partly the learning rate, not just the checkpoint.
4. **Specificity.**
   > The decision models compared with the NLI and noise controls. NLI is the better starting point.
5. **Probes.**
   > Before any training, we tested how much relevance information each frozen model already has. That predicts which starting points end up better, with a rank correlation of 0.90, so it's a cheap way to choose a starting point.

---

## Closing (after the video, about 20 s)

> To sum up: decision models give a small early head start that disappears with more data. Most of that head start isn't specific to decision training, and they can be unstable at high learning rates. The bigger lessons are about method: tune each model's learning rate, use several seeds, and watch for collapsed runs. Everything (the code, the logs and every score) is public on GitHub, and we've written it up as an IEEE paper.

---

## Two-minute explainer (if asked to explain more)

- **Reranker.** A model that takes a (query, passage) pair and outputs a relevance score. We train it with a listwise loss: per query, one correct and seven BM25 "hard" wrong passages, and a softmax that rewards putting the correct one first.
- **Why controls?** If decision models win, it could be because (a) any supervised training helps, which the NLI control tests, or (b) any change of weights helps, which the noise control tests. Without controls, you can't credit "decision training".
- **Why per-model learning rates?** A fair comparison gives every model its best settings. We picked each model's rate on a separate dev set, never the test set.
- **Why multiple seeds?** Fine-tuning is noisy. One run can be lucky or collapse, so we ran 2–3 seeds where it mattered most.
- **Preregistration.** We wrote down the exact test and code before seeing the results and froze it in a git tag. Changes after that are listed as amendments in the paper.
- **Test sets.** 1,000 held-out MS MARCO queries (MRR@10, where 1 means the answer is always first), and TREC Deep Learning 2019+2020 with expert relevance judgments (nDCG@10).
- **Compute.** 95 runs, 4.1 GPU-hours on one RTX 3090.

## Likely questions

**"Inconclusive" sounds like it failed. Is that a result?**
Yes. We said in advance what would count as evidence, and the data didn't reach it. That's honest, and it's useful: anyone who assumed decision models are better starting points now knows the gain is small, temporary and not specific to decision training.

**Why did some runs collapse?**
All five ran at higher learning rates than plain ModernBERT uses. The rate was chosen on one short run, and on a different seed or a longer run it pushed the model off track. Plain ModernBERT never ran at those rates, so we can't say whether the checkpoint or the rate is to blame. That's stated as a limitation.

**Isn't excluding collapsed runs cherry-picking?**
That's why the preregistered result keeps them in and is reported first. The exclusion analysis is labelled exploratory, uses an objective rule (the final loss stays near random guessing), and we apply the same rule to the controls.

**Why only ModernBERT-large?**
It's the only model with five independent public decision checkpoints. Using one backbone means the starting points differ only in their weights, which keeps the comparison fair.

**How is this different from existing work?**
Earlier work tested decision models directly as rerankers, where they lose. We test them as starting points for training a reranker, which nobody had done.

**What would you do next?**
Try more random-noise draws, to see how much of the head start comes from just perturbing the weights. Also pick learning rates that are stable across seeds, and test other backbones.

**Can someone reproduce this?**
Yes. The GitHub repo has the config, code, tests, every run's log and per-query scores. `arbiter analyze` and `arbiter plot` regenerate every number and figure in the paper.
