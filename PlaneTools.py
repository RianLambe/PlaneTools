bl_info = {
    "name": "Plane Tools",
    "author": "Rian Lambe",
    "version": (1, 1),
    "blender": (3, 0, 0),
    "location": "View3D > UI > Plane Tools",
    "description": "Tools for level creation using planes",
    "category": "Object",
}

import bpy
import math
import numpy as np
from bpy.types import Panel, Operator
from bpy.props import FloatProperty, EnumProperty, IntProperty
import mathutils
import bmesh

def get_image_alpha_bounds(image):
    """Get the bounds of non-zero alpha pixels in the image."""
    if not image or not image.pixels:
        return None
    
    # Convert image pixels to numpy array
    pixels = np.array(image.pixels[:])
    width = image.size[0]
    height = image.size[1]
    
    # Reshape array to separate RGBA channels
    pixels = pixels.reshape(height, width, 4)
    
    # Get alpha channel
    alpha = pixels[:, :, 3]
    
    # Find rows and columns with non-zero alpha
    rows = np.any(alpha > 0, axis=1)
    cols = np.any(alpha > 0, axis=0)
    
    if not np.any(rows) or not np.any(cols):
        return None
    
    # Get bounds
    min_row, max_row = np.where(rows)[0][[0, -1]]
    min_col, max_col = np.where(cols)[0][[0, -1]]
    
    # Convert to UV coordinates (0 to 1)
    return {
        'min_u': min_col / width,
        'max_u': (max_col + 1) / width,
        'min_v': min_row / height,
        'max_v': (max_row + 1) / height
    }

class OBJECT_OT_crop_planes_to_alpha(Operator):
    """Crop selected planes to their alpha bounds"""
    bl_idname = "object.crop_planes_to_alpha"
    bl_label = "Crop Images"
    bl_options = {'REGISTER', 'UNDO'}
    
    padding: FloatProperty(
        name="Padding",
        description="Extra padding around the cropped area (in percent)",
        default=0.0,
        min=0.0,
        max=100.0
    )

    @classmethod
    def poll(cls, context):
        return any(obj for obj in context.selected_objects if (
            obj.type == 'MESH' and 
            obj.data.uv_layers and 
            len(obj.data.materials) > 0 and
            obj.data.materials[0].node_tree and
            obj.data.materials[0].node_tree.nodes.get('Image Texture')
        ))

    def execute(self, context):
        # Ensure we're in object mode
        if context.active_object and context.active_object.mode != 'OBJECT':
            bpy.ops.object.mode_set(mode='OBJECT')

        # Store initial state
        original_active = context.active_object
        original_selection = context.selected_objects[:]
        original_cursor = context.scene.cursor.location.copy()
        
        # First pass: Process all geometry
        for obj in original_selection:
            if not (obj.type == 'MESH' and 
                   obj.data.uv_layers and 
                   len(obj.data.materials) > 0 and
                   obj.data.materials[0].node_tree and
                   obj.data.materials[0].node_tree.nodes.get('Image Texture')):
                continue

            mesh = obj.data
            material = obj.data.materials[0]
            nodes = material.node_tree.nodes
            image_node = nodes.get('Image Texture')
            
            if not image_node or not image_node.image:
                continue
            
            # Get image bounds
            bounds = get_image_alpha_bounds(image_node.image)
            if not bounds:
                continue

            # Check if already cropped
            current_uv_bounds = {
                'min_u': min(uv.uv.x for uv in mesh.uv_layers.active.data),
                'max_u': max(uv.uv.x for uv in mesh.uv_layers.active.data),
                'min_v': min(uv.uv.y for uv in mesh.uv_layers.active.data),
                'max_v': max(uv.uv.y for uv in mesh.uv_layers.active.data)
            }
            
            margin = 0.01
            if (abs(current_uv_bounds['min_u'] - bounds['min_u']) < margin and
                abs(current_uv_bounds['max_u'] - bounds['max_u']) < margin and
                abs(current_uv_bounds['min_v'] - bounds['min_v']) < margin and
                abs(current_uv_bounds['max_v'] - bounds['max_v']) < margin):
                continue

            # Apply padding
            padding = self.padding / 100
            bounds['min_u'] = max(0.0, bounds['min_u'] - padding)
            bounds['max_u'] = min(1.0, bounds['max_u'] + padding)
            bounds['min_v'] = max(0.0, bounds['min_v'] - padding)
            bounds['max_v'] = min(1.0, bounds['max_v'] + padding)
            
            # Store original vertex positions and get bounds
            original_verts = [(vert.co.copy()) for vert in mesh.vertices]
            x_coords = [v.co.x for v in mesh.vertices]
            y_coords = [v.co.y for v in mesh.vertices]
            min_x, max_x = min(x_coords), max(x_coords)
            min_y, max_y = min(y_coords), max(y_coords)
            
            # Calculate UV dimensions
            uv_width = bounds['max_u'] - bounds['min_u']
            uv_height = bounds['max_v'] - bounds['min_v']
            uv_center_x = (bounds['min_u'] + bounds['max_u']) / 2
            uv_center_y = (bounds['min_v'] + bounds['max_v']) / 2
            
            # Calculate scale factors
            original_width = max_x - min_x
            original_height = max_y - min_y
            
            # Calculate the offset from UV space to vertex space
            offset_x = (uv_center_x - 0.5) * original_width
            offset_y = (uv_center_y - 0.5) * original_height
            
            # Update UV coordinates
            uv_layer = mesh.uv_layers.active
            
            # Create vertex to UV mapping
            vert_to_uv = {}
            for face in mesh.polygons:
                for vert_idx, loop_idx in zip(face.vertices, face.loop_indices):
                    if vert_idx not in vert_to_uv:
                        vert_to_uv[vert_idx] = loop_idx
            
            # Update UVs using vertex mapping
            for vert_idx, loop_idx in vert_to_uv.items():
                vert = mesh.vertices[vert_idx]
                # Calculate relative position in mesh
                rel_x = (vert.co.x - min_x) / (max_x - min_x)
                rel_y = (vert.co.y - min_y) / (max_y - min_y)
                
                # Map to new UV coordinates
                uv_layer.data[loop_idx].uv.x = bounds['min_u'] + (rel_x * uv_width)
                uv_layer.data[loop_idx].uv.y = bounds['min_v'] + (rel_y * uv_height)
            
            # Update vertices while maintaining position and scale
            for vert_idx, vertex in enumerate(mesh.vertices):
                original_vert = original_verts[vert_idx]
                vertex.co.x = original_vert.x * uv_width + offset_x
                vertex.co.y = original_vert.y * uv_height + offset_y
                vertex.co.z = original_vert.z
            
            mesh.update()
        
        # Second pass: Set origins
        for obj in original_selection:
            if not obj.type == 'MESH':
                continue
                
            bpy.ops.object.select_all(action='DESELECT')
            obj.select_set(True)
            context.view_layer.objects.active = obj
            
            mesh = obj.data
            
            # Calculate bounds in world space
            world_matrix = obj.matrix_world
            verts_world = [world_matrix @ v.co for v in mesh.vertices]
            min_x = min(v.x for v in verts_world)
            max_x = max(v.x for v in verts_world)
            min_y = min(v.y for v in verts_world)
            max_y = max(v.y for v in verts_world)
            min_z = min(v.z for v in verts_world)
            max_z = max(v.z for v in verts_world)
            center_x = (min_x + max_x) / 2
            center_y = (min_y + max_y) / 2
            center_z = (min_z + max_z) / 2
            
            # Set origin based on stretch axis
            if context.scene.stretch_axis == 'X':
                cursor_pos = mathutils.Vector((min_x, center_y, obj.location.z))
            elif context.scene.stretch_axis == 'Y':
                cursor_pos = mathutils.Vector((center_x, min_y, obj.location.z))
            else:  # Z axis
                cursor_pos = mathutils.Vector((center_x, center_y, min_z))
            
            context.scene.cursor.location = cursor_pos
            bpy.ops.object.origin_set(type='ORIGIN_CURSOR')
            
            mesh.update()
            obj.update_tag()

        # Restore original state
        context.scene.cursor.location = original_cursor
        bpy.ops.object.select_all(action='DESELECT')
        for obj in original_selection:
            obj.select_set(True)
        context.view_layer.objects.active = original_active
        
        return {'FINISHED'}

class OBJECT_OT_rotate_and_scale_plane(Operator):
    """Rotate selected objects and adjust dimension"""
    bl_idname = "object.rotate_and_scale_plane"
    bl_label = "Rotate and Scale"
    bl_options = {'REGISTER', 'UNDO'}

    def get_scale(self, obj, axis):
        if axis == 'X':
            return obj.scale.x
        elif axis == 'Y':
            return obj.scale.y
        else:
            return obj.scale.z

    def set_scale(self, obj, axis, value):
        if axis == 'X':
            obj.scale.x = value
        elif axis == 'Y':
            obj.scale.y = value
        else:
            obj.scale.z = value

    def execute(self, context):
        angle_deg = context.scene.rotation_angle
        angle_rad = math.radians(angle_deg)
        rotation_axis = context.scene.rotation_axis
        stretch_axis = context.scene.stretch_axis
        
        self.report({'INFO'}, f"Angle (deg): {angle_deg}, Cos({angle_deg}): {math.cos(math.radians(angle_deg))}")
        
        for obj in context.selected_objects:
            bpy.ops.object.transform_apply(location=False, rotation=False, scale=True)
            
            if rotation_axis == 'X':
                obj.rotation_euler.x = angle_rad
            elif rotation_axis == 'Y':
                obj.rotation_euler.y = angle_rad
            elif rotation_axis == 'Z':
                obj.rotation_euler.z = angle_rad
            
            cos_angle = math.cos(math.radians(angle_deg))
            scale_factor = 1 / cos_angle
            
            self.report({'INFO'}, f"Scale factor: {scale_factor}")
            
            self.set_scale(obj, stretch_axis, scale_factor)
            
            obj.scale = obj.scale
            bpy.context.view_layer.update()
            
        return {'FINISHED'}

class VIEW3D_PT_plane_adjustment_panel(Panel):

    bl_space_type = 'VIEW_3D'
    bl_region_type = 'UI'
    bl_category = 'Plane tools'
    bl_label = "Plane tools"

    def draw(self, context):
        layout = self.layout
        
        layout.label(text="Settings:")
        box = layout.box()
        box.prop(context.scene, "rotation_axis", text="Rotation Axis")
        box.prop(context.scene, "stretch_axis", text="Stretch Axis")
        box.prop(context.scene, "rotation_angle", text="Rotation Angle")
        box.prop(context.scene, "pixels_per_meter", text="Pixels per meter")

        layout.label(text="Operations:")
        box = layout.box()
        box.operator("object.crop_planes_to_alpha")
        box.operator("object.rotate_and_scale_plane")

        # Height adjustment section
        box.separator()
        box.label(text="Adjust Height:")
        grid = box.grid_flow(row_major=True, columns=2, align=True)
        grid.operator("object.adjust_height", text="-1").adjustment = -1
        grid.operator("object.adjust_height", text="+1").adjustment = 1
        grid.operator("object.adjust_height", text="-10").adjustment = -10
        grid.operator("object.adjust_height", text="+10").adjustment = 10

        # Stairs section
        box.separator()
        box.label(text="Stairs:")
        box.prop(context.scene, "num_steps", text="Number of steps")
        box.operator("object.create_stairs")

class OBJECT_OT_adjust_height(Operator):
    """Adjust object height"""
    bl_idname = "object.adjust_height"
    bl_label = "Adjust Height"
    bl_options = {'REGISTER', 'UNDO'}
    
    adjustment: FloatProperty(
        name="Adjustment",
        description="Amount to adjust height in pixels",
        default=1.0,
    )
    
    def execute(self, context):
        # Convert pixel adjustment to meters based on pixels_per_meter setting
        pixels_per_meter = context.scene.pixels_per_meter
        meter_adjustment = self.adjustment / pixels_per_meter
        
        for obj in context.selected_objects:
            if obj.type == 'MESH':
                # Move the object up or down
                obj.location.z += meter_adjustment
        return {'FINISHED'}

class OBJECT_OT_create_stairs(Operator):
    """Create stairs by adding loop cuts"""
    bl_idname = "object.create_stairs"
    bl_label = "Create Stairs"
    bl_options = {'REGISTER', 'UNDO'}
    
    def execute(self, context):
        num_steps = context.scene.num_steps
        
        # Get selected objects
        for obj in context.selected_objects:
            if obj.type != 'MESH':
                continue
                
            # Enter edit mode for this object
            bpy.context.view_layer.objects.active = obj
            bpy.ops.object.mode_set(mode='EDIT')
            
            # Select all edges
            bpy.ops.mesh.select_all(action='SELECT')
            
            # Add all loop cuts at once
            bpy.ops.mesh.loopcut_slide(
                MESH_OT_loopcut={
                    "number_cuts": num_steps - 1,  # -1 because n steps need n-1 cuts
                    "object_index": 0,
                    "edge_index": 2,
                    "mesh_select_mode_init": (True, False, False),
                },
                TRANSFORM_OT_edge_slide={
                    "value": 0.0,  # This will place them evenly
                    "mirror": False,
                    "snap": False,
                    "correct_uv": True
                }
            )
            
            # Switch to face select mode
            bpy.ops.mesh.select_mode(type='FACE')
            
            # Deselect all faces
            bpy.ops.mesh.select_all(action='DESELECT')
            
            # Get mesh data in edit mode
            bm = bmesh.from_edit_mesh(obj.data)
            
            # Sort faces by position (assuming they're ordered from front to back)
            sorted_faces = sorted(bm.faces[:], key=lambda f: f.calc_center_median().y)
            
            # Select every second face
            for i in range(len(sorted_faces)):
                if i % 2 == 1:  # odd indices (0-based)
                    sorted_faces[i].select = True
            
            # Update the mesh
            bmesh.update_edit_mesh(obj.data)
            
            # Scale the selected faces to 0 in Z direction
            bpy.ops.transform.resize(value=(1, 1, 0))
            
            # Return to object mode
            bpy.ops.object.mode_set(mode='OBJECT')
            
        return {'FINISHED'}

def register():
    bpy.types.Scene.rotation_angle = FloatProperty(
        name="Rotation Angle",
        description="Angle to rotate the plane in degrees",
        default=45.0,
        min=-360.0,
        max=360.0
    )
    
    bpy.types.Scene.rotation_axis = EnumProperty(
        name="Rotation Axis",
        description="Axis to rotate around",
        items=[
            ('X', "X Axis", "Rotate around X axis"),
            ('Y', "Y Axis", "Rotate around Y axis"),
            ('Z', "Z Axis", "Rotate around Z axis"),
        ],
        default='X'
    )
    
    bpy.types.Scene.stretch_axis = EnumProperty(
        name="Stretch Axis",
        description="Axis to stretch",
        items=[
            ('X', "X Axis", "Stretch along X axis"),
            ('Y', "Y Axis", "Stretch along Y axis"),
            ('Z', "Z Axis", "Stretch along Z axis"),
        ],
        default='Y'
    )

    bpy.types.Scene.pixels_per_meter = FloatProperty(
        name="Pixels per meter",
        description="Number of pixels that represent one meter",
        default=32.0,
        min=0.1,
        soft_max=1000.0
    )
    
    bpy.types.Scene.num_steps = IntProperty(
        name="Number of Steps",
        description="Number of steps to create",
        default=5,
        min=2,
        max=100
    )
    
    bpy.utils.register_class(OBJECT_OT_crop_planes_to_alpha)
    bpy.utils.register_class(OBJECT_OT_rotate_and_scale_plane)
    bpy.utils.register_class(VIEW3D_PT_plane_adjustment_panel)
    bpy.utils.register_class(OBJECT_OT_adjust_height)
    bpy.utils.register_class(OBJECT_OT_create_stairs)

def unregister():
    del bpy.types.Scene.rotation_angle
    del bpy.types.Scene.rotation_axis
    del bpy.types.Scene.stretch_axis
    
    bpy.utils.unregister_class(VIEW3D_PT_plane_adjustment_panel)
    bpy.utils.unregister_class(OBJECT_OT_rotate_and_scale_plane)
    bpy.utils.unregister_class(OBJECT_OT_crop_planes_to_alpha)
    del bpy.types.Scene.pixels_per_meter
    del bpy.types.Scene.num_steps
    bpy.utils.unregister_class(OBJECT_OT_adjust_height)
    bpy.utils.unregister_class(OBJECT_OT_create_stairs)

if __name__ == "__main__":
    try:
        unregister()
    except:
        pass
    register()