"""Cell/axon drawing helpers, extracted verbatim from the MorphoNAS engine's
main.py (functions _draw_cell_borders_and_axons + _draw_axon_on_axis).

Pulled into the package so the growth animation does not need to import main.py,
whose top-level imports drag in the whole GA/CMA runner stack (cma, etc.) that a
tutorial has no use for. Behaviour is identical to the engine's own rendering.
"""
import numpy as np


def _draw_cell_borders_and_axons(ax, grid, img_data, scale=1.0):
    """Draw cell borders and axons on a given axis (verbatim from engine main.py)."""
    import matplotlib.patches as patches
    import matplotlib.path as mpath

    # Draw neuron connections first (thicker solid lines)
    source_indices, target_indices = grid.neuron_connections.nonzero()
    neuron_ids = grid.get_neuron_ids()
    for source_id, target_id in zip(source_indices + 1, target_indices + 1):
        if source_id in neuron_ids and target_id in neuron_ids:
            source_pos = grid.get_cell_position(source_id)
            target_pos = grid.get_cell_position(target_id)

            # Get the RGB color at source cell's position for the connection color
            source_color = img_data[source_pos]
            connection_color = 1 - source_color  # Invert RGB values
            connection_color = np.clip(connection_color, 0, 1)

            # Calculate shortest path considering wrapping
            dx = target_pos[1] - source_pos[1]
            dy = target_pos[0] - source_pos[0]

            # Adjust for wrapping
            if dx > grid.size_x / 2:
                dx -= grid.size_x
            elif dx < -grid.size_x / 2:
                dx += grid.size_x

            if dy > grid.size_y / 2:
                dy -= grid.size_y
            elif dy < -grid.size_y / 2:
                dy += grid.size_y

            # Calculate the endpoint considering the wrapping
            end_x = source_pos[1] + dx
            end_y = source_pos[0] + dy

            # Draw connection line using the calculated shortest path
            path_vertices = [
                (source_pos[1], source_pos[0]),  # Start point
                (end_x, end_y)  # End point using wrapped coordinates
            ]
            path = mpath.Path(path_vertices, [mpath.Path.MOVETO, mpath.Path.LINETO])
            patch = patches.PathPatch(path, facecolor='none', edgecolor=connection_color,
                                      linewidth=2.0 * scale, alpha=0.9)
            ax.add_patch(patch)

    # Draw cells and growing axons
    for cell_id in grid.get_cell_ids():
        x, y = grid.get_cell_position(cell_id)

        # Get the RGB color at this cell's position
        cell_color = img_data[x, y]
        border_color = 1 - cell_color  # Invert RGB values
        border_color = np.clip(border_color, 0, 1)

        # Set linestyle based on cell type
        linestyle = ':' if grid.is_progenitor(cell_id) else '-'

        # Add cell border
        rect = patches.Rectangle(
            (y - 0.5, x - 0.5), 1, 1,
            linewidth=1.5 * scale,
            edgecolor=border_color,
            facecolor='none',
            linestyle=linestyle
        )
        ax.add_patch(rect)

        # Draw axon if this is a neuron with a growing axon (not yet connected)
        if (grid.is_neuron(cell_id) and
                len(grid.get_axon(cell_id)) > 1 and
                grid.neuron_connections[cell_id - 1].nnz == 0):
            _draw_axon_on_axis(ax, grid.get_axon(cell_id), border_color, grid, scale)


def _draw_axon_on_axis(ax, axon_points, color, grid, scale=1.0):
    """Draw an axon as a dotted curve on a given axis (verbatim from engine main.py)."""
    import matplotlib.patches as patches
    import matplotlib.path as mpath

    if len(axon_points) < 2:
        return

    # Convert axon points to display coordinates (center of cells)
    points = []
    for i in range(len(axon_points)):
        x, y = axon_points[i]

        if i > 0:
            # Get previous point
            prev_x, prev_y = axon_points[i - 1]

            # Calculate differences considering wrapping
            dx = x - prev_x
            dy = y - prev_y

            # Adjust for wrapping
            if dx > grid.size_x / 2:
                dx -= grid.size_x
            elif dx < -grid.size_x / 2:
                dx += grid.size_x

            if dy > grid.size_y / 2:
                dy -= grid.size_y
            elif dy < -grid.size_y / 2:
                dy += grid.size_y

            # Add to the accumulated position
            last_x, last_y = points[-1]
            points.append((last_x + dx, last_y + dy))
        else:
            # First point - swap x,y for matplotlib's coordinate system
            points.append((y, x))

    # Create a smooth path through the points
    path_vertices = []
    codes = []

    # Start point
    path_vertices.append(points[0])
    codes.append(mpath.Path.MOVETO)

    # Add curved segments between points
    for i in range(1, len(points)):
        curr = np.array(points[i])
        prev = np.array(points[i - 1])

        # Calculate control points for the curve
        dist = curr - prev
        ctrl1 = prev + dist * 0.3
        ctrl2 = curr - dist * 0.3

        path_vertices.extend([ctrl1, ctrl2, curr])
        codes.extend([mpath.Path.CURVE4] * 3)

    # Create and add the path
    path = mpath.Path(path_vertices, codes)
    patch = patches.PathPatch(path, facecolor='none', edgecolor=color,
                              linewidth=1.5 * scale, alpha=1, linestyle=':')
    ax.add_patch(patch)
