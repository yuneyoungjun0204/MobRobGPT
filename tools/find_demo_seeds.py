"""데모 영상용 '깨끗한' 시드를 찾는다 — 전멸 포획 · 돌파 0 · 아군 충돌 0.

왜 필요한가
    영상은 방법을 보여주는 것이라, 하필 사고가 난 시드를 고르면 잘못된 인상을 준다.
    그렇다고 결과를 미화하려는 것은 아니다 — **성능 주장은 논문의 30시드 통계로 하고,
    영상은 "정상 동작이 어떤 모습인가"를 보이는 용도**다. 어느 시드를 골랐는지
    파일명과 이 스크립트 출력에 남겨 재현 가능하게 한다.

렌더링 없이 시뮬만 돌리므로 빠르다(에피소드당 수 초). 지휘관은 기본이 휴리스틱 —
LLM 은 호출당 1.5s 라 30시드 x 3대형을 돌리면 비싸다. 휴리스틱으로 후보를 좁힌 뒤
--backend 로 상위 몇 개만 확인하는 순서를 권한다.

사용:
    python tools/find_demo_seeds.py                          # 3대형 x 시드 0..29
    python tools/find_demo_seeds.py --seeds 40 --top 5
    python tools/find_demo_seeds.py --formations wave --seeds 60
    python tools/find_demo_seeds.py --backend gemini --model gemini-3.5-flash-lite --only 3,7,11
"""
from __future__ import annotations

import argparse
import os
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)

FORMATIONS = ("concentrated", "diversionary", "wave")


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="데모용 깨끗한 시드 탐색 (U-Net 기동)")
    p.add_argument("--formations", default=",".join(FORMATIONS))
    p.add_argument("--seeds", type=int, default=30, help="시드 0..N-1 (기본 30)")
    p.add_argument("--only", default=None, help="이 시드들만 (쉼표 구분)")
    p.add_argument("--steps", type=int, default=900, help="에피소드 최대 step")
    p.add_argument("--ckpt", default="boatattack_sim/models/u-net_map.pt")
    p.add_argument("--nets", type=int, default=3)
    p.add_argument("--replan", type=int, default=60)
    p.add_argument("--backend", default=None, help="지정하면 LLM 지휘관 (느림)")
    p.add_argument("--model", default=None)
    p.add_argument("--command", default="Capture all enemies")
    p.add_argument("--top", type=int, default=3, help="대형별 상위 N개 출력")
    p.add_argument("--no-commander", action="store_true",
                   help="지휘관 계층을 빼고 시뮬 내장 배정으로만 돈다"
                        "(make_commander_gif.py --no-commander 와 같은 구성)")
    p.add_argument("--site", default=None,
                   help="해역(지형). 해역이 바뀌면 육지 마스크가 달라져 같은 시드도 결과가 "
                        "달라진다 — 지형별 영상을 만들려면 해역마다 다시 찾아야 한다")
    return p.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    import matplotlib
    matplotlib.use("Agg")
    import numpy as np

    from commander.fallback import heuristic_plan
    from commander.rl_bridge import build_battlefield_defense
    from commander.sim_bridge import plan_to_assign
    from commander.unet_bridge import CommandedCnnEnv

    commander = None
    if args.backend:
        try:
            from dotenv import load_dotenv
            load_dotenv(os.path.join(REPO, ".env"))
        except ImportError:
            pass
        from commander import make_commander
        commander = make_commander(args.backend, args.model)
        commander.warmup()

    seeds = ([int(x) for x in args.only.split(",") if x.strip()] if args.only
             else list(range(args.seeds)))
    ckpt = args.ckpt if os.path.isabs(args.ckpt) else os.path.join(REPO, args.ckpt)
    who = args.model or args.backend or "heuristic"
    from tools.make_commander_gif import resolve_site
    lat, lon, site_label = resolve_site(args.site)
    geo = None if lat is None else (lat, lon)

    print(f"[seeds] U-Net={os.path.basename(ckpt)} 지휘관={who} 해역={site_label} "
          f"시드 {len(seeds)}개 x 대형 {len(args.formations.split(','))}개")
    results: dict[str, list[dict]] = {}

    for form in [f.strip() for f in args.formations.split(",") if f.strip()]:
        env = CommandedCnnEnv(ckpt, enemy_mode=form, nets_per_ship=args.nets, geo=geo)
        if args.no_commander:
            env.assign_source = "heuristic"      # 계획 주입 없음 — 시뮬이 스스로 배정
        rows = []
        for sd in seeds:
            env.reset(seed=sd)
            env.running = True
            last = -10 ** 9
            for _ in range(args.steps):
                t = int(env.t.flat[0])
                if not args.no_commander and t - last >= args.replan:
                    bf = build_battlefield_defense(env, command=args.command)
                    plan = commander.plan(bf) if commander else heuristic_plan(bf)
                    plan_to_assign(plan, bf, mode="llm")
                    env.set_plan(plan, args.command)
                    last = t
                if bool(env.done.flat[0]):
                    break
                env.step()
            st = dict(env.stats)
            cap = int(st.get("captures", 0))
            brc = int(st.get("breaches", 0))
            col = int(st.get("ally_collisions", 0))
            n_e = int(env.cfg.n_enemies)
            # ★ 아군 손실은 stats 에 없다 — 살아있는 아군을 직접 센다.
            #   충돌 0 이어도 다른 이유로 비활성화될 수 있으므로 따로 확인한다.
            lost = int((~np.asarray(env.a_alive[0], bool)).sum())
            full = cap >= n_e
            rows.append({"seed": sd, "cap": cap, "brc": brc, "col": col, "lost": lost,
                         "nets": int(st.get("nets_used", 0)),
                         "steps": int(env.t.flat[0]), "n_enemies": n_e,
                         "clean": brc == 0 and col == 0 and lost == 0 and full,
                         "frac": cap / n_e})
            mark = "OK" if rows[-1]["clean"] else "  "
            print(f"  {form:<14s} seed {sd:<3d} 포획 {cap:2d}/{n_e}  돌파 {brc}  "
                  f"충돌 {col}  아군손실 {lost}  그물 {rows[-1]['nets']:2d}  "
                  f"t={rows[-1]['steps']:4d}  [{mark}]", flush=True)
        results[form] = rows

    print("\n" + "=" * 74)
    print("대형별 추천 (전멸 포획 · 돌파 0 · 충돌 0 · 아군손실 0, 빨리 끝난 순)")
    print("=" * 74)
    best = {}
    for form, rows in results.items():
        clean = [r for r in rows if r["clean"]]
        clean.sort(key=lambda r: (-r["cap"], r["steps"]))
        if not clean:
            print(f"  {form:<14s} 조건을 만족하는 시드 없음 — --seeds 를 늘려보라")
            continue
        best[form] = clean[0]["seed"]
        for r in clean[: args.top]:
            print(f"  {form:<14s} seed {r['seed']:<3d} 포획 {r['cap']}/{r['n_enemies']}  "
                  f"t={r['steps']}  그물 {r['nets']}")
    if best:
        print("\n[seeds] 렌더 명령:")
        for form, sd in best.items():
            print(f"  python tools/make_commander_gif.py --enemy {form} --seed {sd} "
                  f"--frames {min(900, 1 + max(r['steps'] for r in results[form]))} --zoom")
    return 0


if __name__ == "__main__":
    sys.exit(main())
