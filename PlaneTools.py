#Blender plugin info
bl_info = {
    "name": "Plane Tools",
    "author": "Rian Lambe",
    "version": (1, 1),
    "blender": (3, 0, 0),
    "location": "View3D > UI > Plane Tools",
    "description": "Tools for level creation using planes",
    "category": "Object",
}

#Imports 
import bpy
import math
import numpy as np
from bpy.types import Panel, Operator
from bpy.props import FloatProperty, EnumProperty, IntProperty
import mathutils
import bmesh

#Crop selected planes to their alpha bounds
class CropImages(Operator):

    #Panel registration properties
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

    #Checks if objects are selected
    @classmethod
    def poll(cls, context):
        return any(obj for obj in context.selected_objects if (
            obj.type == 'MESH' and 
            obj.data.uv_layers and 
            len(obj.data.materials) > 0 and
            obj.data.materials[0].node_tree and
            obj.data.materials[0].node_tree.nodes.get('Image Texture')
        )
    )

    #Calculates the bounds of the image based on the alpha
    def GetImageBounds(Image):
    
        #Check valid image passed
        if not Image or not Image.pixels:
            return None
        
        # Convert image pixels to numpy array
        Pixels = np.array(Image.pixels[:])
        Width = Image.size[0]
        Height = Image.size[1]
        
        # Reshape array to separate RGBA channels
        Pixels = Pixels.reshape(Height, Width, 4)
        
        # Get alpha channel
        Alpha = Pixels[:, :, 3]
        
        # Find rows and columns with non-zero alpha
        Rows = np.any(Alpha > 0, axis=1)
        Cols = np.any(Alpha > 0, axis=0)
        
        if not np.any(Rows) or not np.any(Cols):
            return None
        
        # Get bounds
        MinRow, MaxRow = np.where(Rows)[0][[0, -1]]
        MinCol, MaxCol = np.where(Cols)[0][[0, -1]]
        
        # Convert to UV coordinates (0 to 1)
        return {
            'min_u': MinCol / Width,
            'max_u': (MaxCol + 1) / Width,
            'min_v': MinRow / Height,
            'max_v': (MaxRow + 1) / Height
        }

    def execute(self, context):
        #Check in object mode
        if context.active_object and context.active_object.mode != 'OBJECT':
            bpy.ops.object.mode_set(mode='OBJECT')

        #Store initial state
        original_active = context.active_object
        original_selection = context.selected_objects[:]
        original_cursor = context.scene.cursor.location.copy()
        
        #Process all geometry
        for Obj in original_selection:
            if not (Obj.type == 'MESH' and 
                   Obj.data.uv_layers and 
                   len(Obj.data.materials) > 0 and
                   Obj.data.materials[0].node_tree and
                   Obj.data.materials[0].node_tree.nodes.get('Image Texture')):
                continue

            mesh = Obj.data
            material = Obj.data.materials[0]
            nodes = material.node_tree.nodes
            image_node = nodes.get('Image Texture')
            
            if not image_node or not image_node.image:
                continue
            
            #Get image bounds
            Bounds = get_image_alpha_bounds(image_node.image)
            if not Bounds:
                continue

            #Check if already cropped
            CurrentUVBounds = {
                'min_u': min(uv.uv.x for uv in mesh.uv_layers.active.data),
                'max_u': max(uv.uv.x for uv in mesh.uv_layers.active.data),
                'min_v': min(uv.uv.y for uv in mesh.uv_layers.active.data),
                'max_v': max(uv.uv.y for uv in mesh.uv_layers.active.data)
            }
            
            Margin = 0.01
            if (abs(CurrentUVBounds['min_u'] - Bounds['min_u']) < Margin and
                abs(CurrentUVBounds['max_u'] - Bounds['max_u']) < Margin and
                abs(CurrentUVBounds['min_v'] - Bounds['min_v']) < Margin and
                abs(CurrentUVBounds['max_v'] - Bounds['max_v']) < Margin):
                continue

            #Apply padding
            Padding = self.padding / 100
            Bounds['min_u'] = max(0.0, Bounds['min_u'] - Padding)
            Bounds['max_u'] = min(1.0, Bounds['max_u'] + Padding)
            Bounds['min_v'] = max(0.0, Bounds['min_v'] - Padding)
            Bounds['max_v'] = min(1.0, Bounds['max_v'] + Padding)
            
            # Store original vertex positions and get bounds
            original_verts = [(vert.co.copy()) for vert in mesh.vertices]
            x_coords = [v.co.x for v in mesh.vertices]
            y_coords = [v.co.y for v in mesh.vertices]
            min_x, max_x = min(x_coords), max(x_coords)
            min_y, max_y = min(y_coords), max(y_coords)
            
            # Calculate UV dimensions
            uv_width = Bounds['max_u'] - Bounds['min_u']
            uv_height = Bounds['max_v'] - Bounds['min_v']
            uv_center_x = (Bounds['min_u'] + Bounds['max_u']) / 2
            uv_center_y = (Bounds['min_v'] + Bounds['max_v']) / 2
            
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
                uv_layer.data[loop_idx].uv.x = Bounds['min_u'] + (rel_x * uv_width)
                uv_layer.data[loop_idx].uv.y = Bounds['min_v'] + (rel_y * uv_height)
            
            # Update vertices while maintaining position and scale
            for vert_idx, vertex in enumerate(mesh.vertices):
                original_vert = original_verts[vert_idx]
                vertex.co.x = original_vert.x * uv_width + offset_x
                vertex.co.y = original_vert.y * uv_height + offset_y
                vertex.co.z = original_vert.z
            
            mesh.update()
        
        # Second pass: Set origins
        for Obj in original_selection:
            if not Obj.type == 'MESH':
                continue
                
            bpy.ops.object.select_all(action='DESELECT')
            Obj.select_set(True)
            context.view_layer.objects.active = Obj
            
            mesh = Obj.data
            
            # Calculate bounds in world space
            world_matrix = Obj.matrix_world
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
                cursor_pos = mathutils.Vector((min_x, center_y, Obj.location.z))
            elif context.scene.stretch_axis == 'Y':
                cursor_pos = mathutils.Vector((center_x, min_y, Obj.location.z))
            else:  # Z axis
                cursor_pos = mathutils.Vector((center_x, center_y, min_z))
            
            context.scene.cursor.location = cursor_pos
            bpy.ops.object.origin_set(type='ORIGIN_CURSOR')
            
            mesh.update()
            Obj.update_tag()

        # Restore original state
        context.scene.cursor.location = original_cursor
        bpy.ops.object.select_all(action='DESELECT')
        for Obj in original_selection:
            Obj.select_set(True)
        context.view_layer.objects.active = original_active
        
        return {'FINISHED'}

#Rotate selected objects and adjust scale
class RotateAndScale(Operator):
    
    #Panel registration properties
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

#Adjust heigt based on pixels per meter and angle
class AdjustHeight(Operator):
    
    #Panel registration properties
    bl_idname = "object.adjust_height"
    bl_label = "Adjust Height"
    bl_options = {'REGISTER', 'UNDO'}
    
    adjustment: FloatProperty(
        name="Adjustment",
        description="Amount to adjust height in pixels",
        default=1.0,
    )
    
    def execute(self, context):
        # Convert pixel adjustment to meters based on PixelsPerMeter setting
        PixelsPerMeter = context.scene.pixels_per_meter
        MeterAdjustment = self.adjustment / PixelsPerMeter
        
        for obj in context.selected_objects:
            if obj.type == 'MESH':
                # Move the object up or down
                obj.location.z += MeterAdjustment
        return {'FINISHED'}

#Generates stairs 
class CreateStairs(Operator):
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

#Create blender window
class RenderPlaneToolsPanel(Panel):

    #Panel registration properties
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'UI'
    bl_category = 'Plane tools'
    bl_label = "Plane tools"

    #Draw window in 3D view sidebar
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

#Adds custom properties to Blender's Scene type and registers classes
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
    
    bpy.utils.register_class(CropImages)
    bpy.utils.register_class(RotateAndScale)
    bpy.utils.register_class(RenderPlaneToolsPanel)
    bpy.utils.register_class(AdjustHeight)
    bpy.utils.register_class(CreateStairs)

#Removes custom properties and unregisters classes when disabling the add-on
def unregister():
    del bpy.types.Scene.rotation_angle
    del bpy.types.Scene.rotation_axis
    del bpy.types.Scene.stretch_axis
    
    bpy.utils.unregister_class(RenderPlaneToolsPanel)
    bpy.utils.unregister_class(RotateAndScale)
    bpy.utils.unregister_class(CropImages)
    del bpy.types.Scene.pixels_per_meter
    del bpy.types.Scene.num_steps
    bpy.utils.unregister_class(AdjustHeight)
    bpy.utils.unregister_class(CreateStairs)

#Setup main 
if __name__ == "__main__":
    try:
        unregister()
    except:
        pass
    register()