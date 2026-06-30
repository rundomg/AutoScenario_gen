"""把 CARLA 的观察视角(spectator)定位到某次重建结果锁定的区域。

用法:
    python view_scene_in_carla.py results/auto_result_20260627_184920
    python view_scene_in_carla.py results/auto_result_20260627_184920 --scene s0000_c0
    python view_scene_in_carla.py <folder> --view chase   # 第三人称跟在 ego 后面
    python view_scene_in_carla.py <folder> --no-spawn      # 只移视角,不放占位车

读取该结果文件夹里的 {scene}_match.json(目标世界)和 {scene}_actors.json
(ego / 其它 actor 的真实世界坐标),不写死任何坐标。CARLA 服务器需先启动。
"""

import argparse
import glob
import json
import math
import os
import time

import carla


def _find_scene_id(folder, scene):
    if scene:
        return scene
    matches = sorted(glob.glob(os.path.join(folder, "*_actors.json")))
    if not matches:
        raise SystemExit(f"在 {folder} 找不到 *_actors.json")
    return os.path.basename(matches[0])[: -len("_actors.json")]


def _load_json(path: str):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _world_name(match: dict) -> str:
    # match.json 里是 'Carla/Maps/Town12/Town12' 这种,load_world 只要 'Town12'
    raw = match.get("world_name") or "Town10HD_Opt"
    return os.path.basename(raw)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("folder", help="结果文件夹,如 results/auto_result_20260627_184920")
    parser.add_argument("--scene", default=None, help="场景 id,默认取文件夹里第一个")
    parser.add_argument("--host", default="localhost")
    parser.add_argument("--port", type=int, default=2000)
    parser.add_argument("--timeout", type=float, default=120.0,
                        help="连接/加载超时(秒)。Town12 很大,默认放宽到 120s")
    parser.add_argument("--view", choices=["top", "chase"], default="top",
                        help="top=正上方俯视;chase=ego 后上方第三人称")
    parser.add_argument("--height", type=float, default=60.0, help="俯视相机高度(米)")
    parser.add_argument("--no-spawn", action="store_true",
                        help="不 spawn 占位车,只移动视角")
    parser.add_argument("--hold", type=float, default=120.0,
                        help="脚本结束前保持多少秒(便于观察),0=立即退出")
    args = parser.parse_args()

    scene = _find_scene_id(args.folder, args.scene)
    match = _load_json(os.path.join(args.folder, f"{scene}_match.json"))
    actors = _load_json(os.path.join(args.folder, f"{scene}_actors.json"))
    entities = actors.get("entities", [])
    if not entities:
        raise SystemExit("actors.json 里没有 entities")

    world_name = _world_name(match)
    print(f"[view] 场景={scene}  目标世界={world_name}  actor 数={len(entities)}")

    client = carla.Client(args.host, args.port)
    client.set_timeout(args.timeout)

    cur = client.get_world()
    cur_map = cur.get_map().name  # e.g. 'Carla/Maps/Town12/Town12'
    if world_name.lower() not in cur_map.lower():
        print(f"[view] 当前世界 {cur_map},正在加载 {world_name}(可能要几十秒)...")
        world = client.load_world(world_name)
    else:
        print(f"[view] 已在 {cur_map},跳过重新加载")
        world = cur

    # ego + 全体坐标
    ego = next((e for e in entities if str(e.get("id")) in {"ego", "ego_vehicle"}),
               entities[0])
    locs = [e["location"] for e in entities]
    cx = sum(p["x"] for p in locs) / len(locs)
    cy = sum(p["y"] for p in locs) / len(locs)
    cz = sum(p["z"] for p in locs) / len(locs)
    ego_loc = ego["location"]
    ego_yaw = float((ego.get("rotation") or {}).get("yaw", 0.0))

    # 可选:放半透明占位车,方便看落点(非真实 blueprint,只是 Cybertruck 占位)
    spawned = []
    if not args.no_spawn:
        bp_lib = world.get_blueprint_library()
        bp = bp_lib.filter("vehicle.tesla.cybertruck")[0]
        for e in entities:
            tf = carla.Transform(
                carla.Location(**{k: float(e["location"][k]) for k in ("x", "y", "z")}),
                carla.Rotation(yaw=float((e.get("rotation") or {}).get("yaw", 0.0))),
            )
            a = world.try_spawn_actor(bp, tf)
            if a is not None:
                spawned.append(a)
                print(f"[view] spawn {e.get('id')} @ "
                      f"({tf.location.x:.1f},{tf.location.y:.1f}) yaw={tf.rotation.yaw:.1f}")
            else:
                print(f"[view] !! {e.get('id')} spawn 失败(该点可能不可放置)")

    # debug 标记:各 actor 一个点+文字,路口中心一个大点
    dbg = world.debug
    for e in entities:
        p = e["location"]
        loc = carla.Location(p["x"], p["y"], p["z"] + 1.0)
        is_ego = str(e.get("id")) in {"ego", "ego_vehicle"}
        color = carla.Color(0, 200, 255) if is_ego else carla.Color(255, 80, 80)
        dbg.draw_point(loc, size=0.25, color=color, life_time=args.hold or 120.0)
        dbg.draw_string(loc + carla.Location(z=1.5), str(e.get("id")),
                        color=color, life_time=args.hold or 120.0)

    # 移动 spectator
    spectator = world.get_spectator()
    if args.view == "top":
        cam = carla.Transform(
            carla.Location(x=cx, y=cy, z=cz + args.height),
            carla.Rotation(pitch=-90.0, yaw=ego_yaw),
        )
    else:  # chase:ego 正后方上空
        back = 8.0
        rad = math.radians(ego_yaw)
        cam = carla.Transform(
            carla.Location(
                x=ego_loc["x"] - back * math.cos(rad),
                y=ego_loc["y"] - back * math.sin(rad),
                z=ego_loc["z"] + 4.0,
            ),
            carla.Rotation(pitch=-15.0, yaw=ego_yaw),
        )
    spectator.set_transform(cam)
    print(f"[view] spectator -> ({cam.location.x:.1f},{cam.location.y:.1f},"
          f"{cam.location.z:.1f}) view={args.view}")
    print(f"[view] 区域中心 ≈ ({cx:.1f},{cy:.1f},{cz:.1f})")

    if args.hold > 0:
        print(f"[view] 保持 {args.hold:.0f}s(Ctrl+C 提前退出)...")
        try:
            time.sleep(args.hold)
        except KeyboardInterrupt:
            pass
    for a in spawned:
        try:
            a.destroy()
        except Exception:
            pass


if __name__ == "__main__":
    main()
