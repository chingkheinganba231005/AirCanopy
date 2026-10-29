# Where to run the graphics-card jobs

Prices checked on 2026-10-29. They change often: `uv run python benchmarks/gpu_offers.py`
records a new snapshot of RunPod's and Vast.ai's prices in `results/gpu_offers.json`. Every
source is listed in `docs/sources.md`; conversions use HK$7.8 to the US dollar, the middle
of the Hong Kong Monetary Authority's band of 7.75 to 7.85.

## What the jobs need

- **An NVIDIA card and Linux.** JAX's packages for CUDA 12 work on cards of compute
  capability 5.2 or newer, with a driver of version 525 or newer, on Linux only (JAX
  installation guide). Every card below qualifies.
- **Memory.** Each cell holds 27 packets of 4 bytes in 32-bit numbers: 108 bytes. The
  Taylor–Green run at 256³ cells holds 1.8 GB of packets, plus the step's temporary arrays.
  Everything left in P2 fits on a card with 24 GB. City-sized runs later, with hundreds of
  millions of cells, will need 80 GB or several cards.
- **Time without interruption.** From a quarter of an hour (the speed test) to many hours
  (turbulent averages). A long run needs a machine that keeps going while nobody watches.

## What decides the speed

A lattice Boltzmann step reads every packet of every cell, does a few hundred additions and
multiplications with them, and writes them back. A graphics card can do arithmetic much
faster than it can fetch numbers from its memory, so the step mostly waits for memory: its
speed follows the card's *memory bandwidth*, the bytes it moves per second (Wellein et al.
2006; lesson 6.18 in `JOURNAL.md`). If our step behaves like that, the work a card does per
hour is in proportion to its bandwidth. The speed test checks this: it times both the step
and one plain copy of the packets, and a step that takes only a few copy-times is limited
by bandwidth.

Bandwidth that Vast.ai measured on the machines it lists (median, GB/s; `results/gpu_offers.json`):

| RTX 3090 | RTX 4090 | RTX 5090 | A100 80 GB | H100 |
| --- | --- | --- | --- | --- |
| 805 | 879 | 1455 | 1664 | 2880 (one machine) |

NVIDIA's datasheet gives 2039 GB/s for the A100 80 GB; measured values are lower than
datasheet ones, so only measured values are compared here.

## The options

"Work" is what the money buys in hours of an A100 80 GB, if speed follows bandwidth (hours ×
the card's bandwidth ÷ the A100's). HK$77 is what 100 Colab units cost you.

| Service and card | Memory | Per hour: US$ (HK$) | Hours for HK$77 | Work for HK$77 | Long runs |
| --- | --- | --- | --- | --- | --- |
| Colab, A100 80 GB | 80 GB | 7.52 units (5.8) | 13.3 (you measured about 14) | 13.3 | At most 12 h per runtime, 24 h with Pro+; times out when idle |
| RunPod Community, RTX 3090 | 24 GB | 0.22 (1.7) | 45 | 22 | Runs until you stop it |
| RunPod Community, RTX 4090 | 24 GB | 0.34 (2.7) | 29 | 15 | As above; few in stock at the time |
| RunPod Community, RTX 5090 | 32 GB | 0.69 (5.4) | 14 | 12.5 | As above |
| RunPod Community, A100 80 GB PCIe | 80 GB | 1.19 (9.3) | 8.3 | about 8 | As above |
| RunPod Secure, RTX 4090 | 24 GB | 0.89 (6.9) | 11 | 6 | Data centres vetted by RunPod |
| Vast.ai, RTX 3090 (median of 34 offers; lowest 0.14) | 24 GB | 0.24 (1.9) | 41 | 20 | On-demand machines are not taken back; billed by the second; each host sets its price, and reliability varies by host |
| Vast.ai, RTX 4090 (median of 55; lowest 0.38) | 24 GB | 0.50 (3.9) | 20 | 10.5 | As above |
| Vast.ai, A100 80 GB (median of 8; lowest 1.00) | 80 GB | 1.12 (8.8) | 8.8 | 8.8 | As above |
| Thunder Compute, A100 80 GB | 80 GB | 1.09 (8.5) | 9.1 | about 9 | Its documentation does not mention JAX: test before relying on it |
| Modal, A100 80 GB | 80 GB | 2.50 (19.5) | 4.0 | 4.0 | US$30 of free use every month, 12 h of this card; each call runs up to 24 h; needs a small Modal script |
| Lambda, A100 40 GB (no single 80 GB card listed) | 40 GB | 1.99 (15.5) | 5.0 | – | – |

Notes:

- Google does not publish how many units an hour each Colab card uses. 7.52 units an hour for
  the A100 80 GB is McCormick's measurement of 2026-03-10; with 100 units for HK$77 it gives
  13.3 h, close to the 14 h you found.
- The RTX cards are gaming cards, built for 32-bit numbers. Our long runs use 32-bit
  numbers; the speed test also times 64-bit ones, which may be much slower on these cards.
- Kaggle gives free T4 and P100 cards for about 30 h a week, in sessions of up to 12 h.
  UNVERIFIED: these figures come from third-party summaries; Kaggle does not publish its
  quota.
- Google AI Pro subscriptions are reported to include 200 Colab units a month. UNVERIFIED:
  news reports only.

## Recommendation

1. **For the next jobs** (speed test, Taylor–Green at 256³, the boundary layer on finer
   grids, the cube's approach channel): rent an RTX 4090 on RunPod's Community Cloud, or an
   RTX 3090 when no 4090 is free. For HK$77 a
   4090 runs 29 h, and by bandwidth does a little more work than Colab's 13 h of A100, without
   the 12-hour limit or the idle timeouts. Start with the speed test alone (a quarter of an
   hour, under HK$1): it shows whether the step follows bandwidth.
2. **For runs that need more than 24 GB** (whole districts, later): Colab's A100 80 GB is the
   cheapest 80 GB card per hour in this list. Colab's command-line tool (below) avoids the
   browser; long runs must save checkpoints, because a runtime ends after 12 h. For runs longer
   than Colab allows, an A100 80 GB on Vast.ai or RunPod costs about HK$9 an hour.
3. **Free hours**: Modal's US$30 a month covers 12 h of an A100 80 GB. Worth setting up once
   the runs are routine.

## How to run on RunPod, from an iPad, overnight

The pod runs on RunPod's computers, so the jobs go on after you close the browser and the
iPad sleeps. The pod costs its hourly price from the moment it starts until you terminate it,
even after its jobs have ended. Without a volume, nothing else costs money.

**Evening**

1. Deploy a pod. At the top of the Deploy page choose **Community Cloud** (cheaper than Secure
   Cloud for the same card: on 2026-10-29, US$0.34 an hour for an RTX 4090 against US$0.89),
   and **On-Demand**, not Spot (a Spot pod can be taken back during the night). Choose an RTX
   4090, or an RTX 3090 if no 4090 is free. RunPod's PyTorch template (CUDA 12.8) is fine,
   and so is its container disk of 20 to 50 GB. No volume is needed.
2. When the pod is running, open it, press **Connect**, and open **Jupyter Lab** (then File,
   New, Terminal) or the **web terminal**.
3. Paste this one line and press Return:

   ```
   cd /workspace && curl -fsSLO https://raw.githubusercontent.com/chingkheinganba231005/AirCanopy/main/scripts/gpu_run.sh && nohup setsid bash gpu_run.sh > gpu_run.log 2>&1 < /dev/null &
   ```

4. After a minute, check that it has started:

   ```
   tail -n 20 /workspace/gpu_run.log
   ```

   It shows the card's name, then the repository, then "Python environment", which takes a
   few minutes, then "job 1 of 8". From then on you can close the browser and sleep. `nohup`
   and `setsid` detach the run from the browser's terminal.

**Morning**

5. Open the pod again (Connect, Jupyter Lab). Open `gpu_run.log` in the file list on the left
   and scroll to the end. It should end with `done: 8 jobs, ... failed`.
6. In the file list, open the folder `aircanopy`, long-press `aircanopy_gpu_results.zip`, and
   choose Download. Safari keeps it in the Files app, in Downloads. Send that zip in our chat.
   The zip is rewritten after every job, so it holds every result finished so far, even if a
   job failed or the night was not long enough.
7. Then **terminate** the pod (on the pod's page or its menu in the list of pods). A
   pod that is only stopped erases its container disk, and the zip with it, so download first.

The jobs are listed in `scripts/gpu_jobs.txt`: quick ones first (speed, the Taylor–Green
vortex), then the boundary layer on finer grids. Each job may run for at most 4 hours
(`JOB_HOURS`). To run only the speed test, put the job after the script's name:
`nohup setsid bash gpu_run.sh "benchmarks/wind_speed.py 128 400" > gpu_run.log 2>&1 < /dev/null &`.

With `STOP_POD=1` in front of `nohup`, the script stops the pod when the jobs end, so the card
stops costing money at once. It does so only when `/workspace` is a separate volume (added
when deploying), because stopping erases the container disk. The volume then keeps the zip,
but it stays on that machine, so to download the zip the pod must be started there again,
which needs a free card on it.

## On Vast.ai

The same script and commands. Choose a verified offer with reliability of 0.99 or more and a
CUDA 12 image, open its Jupyter terminal or connect by SSH, and type the commands of step 4
(in your home folder instead of `/workspace`). Destroy the instance when done: a stopped
instance still pays for its disk.

## On Colab, from a terminal

Since June 2026, Colab has a command-line tool (open source, Apache-2.0; Linux and macOS only,
per its README). `colab new --gpu A100` opens a runtime, `colab console` a shell on it, where
the commands of step 4 run the same script; `colab download` fetches files and `colab stop`
ends the runtime. Its README says the high-memory option, which selects the 80 GB A100 in the
notebook interface, needs Colab Pro or Pro+. Not tried yet; whether a run survives the laptop
going to sleep is not documented.
