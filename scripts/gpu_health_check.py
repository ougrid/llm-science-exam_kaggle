"""Manual probe: is a currently-running GPU job actually healthy?

`nvidia-smi` utilization alone cannot tell "computing normally against VRAM"
apart from "computing normally against WSL2's slow shared-system-memory
fallback" -- both look like a busy GPU. This prints the two numbers that
actually distinguish them: memory headroom against the true card size, and
current temperature/power (a job silently running in the slow fallback path
draws much less power than one saturating the card, since it's PCIe-bound,
not compute-bound).

Run anytime: `python scripts/gpu_health_check.py`. Not a substitute for
`src/llmsci/gpu_guard.py`'s startup checks -- this is for eyeballing an
already-running process.
"""

import subprocess

import torch


def main() -> None:
    if not torch.cuda.is_available():
        print("no CUDA device visible")
        return

    props = torch.cuda.get_device_properties(0)
    total_mib = props.total_memory / 1024**2
    print(f"card: {props.name}, true total VRAM: {total_mib:.0f} MiB")

    out = subprocess.check_output(
        [
            "nvidia-smi",
            "--query-gpu=memory.used,utilization.gpu,temperature.gpu,power.draw",
            "--format=csv,noheader,nounits",
        ]
    ).decode().strip()
    used_mib, util, temp, power = [x.strip() for x in out.split(",")]
    used_mib = float(used_mib)
    headroom_pct = 100 * (1 - used_mib / total_mib)

    print(f"memory used: {used_mib:.0f} MiB ({100 * used_mib / total_mib:.1f}% of card, {headroom_pct:.1f}% headroom)")
    print(f"utilization: {util}%   temperature: {temp}C   power draw: {power}W")

    if used_mib > total_mib:
        print("ALARM: reported usage exceeds true card size -- this should be impossible with "
              "gpu_guard.cap_memory_fraction() active; if you see this, the fraction cap is "
              "missing from the running process.")
    elif headroom_pct < 3:
        print("WARNING: <3% headroom -- right at the cliff where the next allocation would "
              "overflow into WSL2's slow shared-memory fallback if the cap isn't active.")
    else:
        print("memory: OK")

    print()
    print("This does not measure throughput. Cross-check against the job's own log: compute "
          "elapsed time (`ps -o etime -p <pid>`) divided by optimizer steps completed, and "
          "compare against the ms/step the run's gpu_guard probe printed at startup -- if the "
          "job is now taking noticeably longer per step than its own startup probe measured, "
          "something degraded mid-run (thermal throttle, contention) even if memory looks fine.")


if __name__ == "__main__":
    main()
