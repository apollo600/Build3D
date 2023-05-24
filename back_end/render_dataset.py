from datasets import dataset_dict
import plotly.graph_objects as go


def visulize_dataset(data_path:str):
    dataset = dataset_dict['phototourism']\
        (data_path, 'train', img_downscale=2, use_cache=True)

    rays_o, rays_d = dataset.all_rays[:10880, :3], dataset.all_rays[:10880, 3:6]
    near, far = dataset.all_rays[:10880, 6:7], dataset.all_rays[:10880, 7:8]

    start = (rays_o + near * rays_d).numpy()
    end = (rays_o + far * rays_d).numpy()

    fig = go.Figure()

    z_in_range = dataset.xyz_world[:, 2]<5
    skip = 5

    fig.add_trace(
        go.Scatter3d(
            x=dataset.xyz_world[z_in_range, 0][::skip],
            y=dataset.xyz_world[z_in_range, 1][::skip],
            z=dataset.xyz_world[z_in_range, 2][::skip],
            mode='markers',
            name='scene',
            marker=dict(size=0.3)
        )
    )

    fig.add_trace(
        go.Scatter3d(
            x=dataset.poses[:, 0, 3],
            y=dataset.poses[:, 1, 3],
            z=dataset.poses[:, 2, 3],
            mode='markers',
            name='cameras',
            marker=dict(size=1)
        )
    )

    xlines = []
    ylines = []
    zlines = []
    for i in [0, 127, -128, -1]:
        xlines += [start[i, 0], end[i, 0], None]
        ylines += [start[i, 1], end[i, 1], None]
        zlines += [start[i, 2], end[i, 2], None]
        
    for p in [[0, 127], [127, -1], [-128, -1], [-128, 0]]:
        xlines += [start[p[0], 0], start[p[1], 0], None]
        ylines += [start[p[0], 1], start[p[1], 1], None]
        zlines += [start[p[0], 2], start[p[1], 2], None]
        xlines += [end[p[0], 0], end[p[1], 0], None]
        ylines += [end[p[0], 1], end[p[1], 1], None]
        zlines += [end[p[0], 2], end[p[1], 2], None]

    fig.add_trace(
        go.Scatter3d(
            x=xlines,
            y=ylines,
            z=zlines,
            mode='lines',
            name='frustum',
            marker=dict(size=1)
        )
    )

    pose = dataset.poses_dict[dataset.img_ids_train[0]]


    fig.add_trace(
        go.Scatter3d(
            x=[pose[0, 3]],
            y=[pose[1, 3]],
            z=[pose[2, 3]],
            mode='markers',
            name='camera',
            marker=dict(size=4)
        )
    )

    # fig.show()
    fig.write_html("../plot.html")


if __name__ == "__main__":
    visulize_dataset("data/brandenburg_gate")