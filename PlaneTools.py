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
from bpy.types import Panel, Operator, PropertyGroup
from bpy.props import FloatProperty, EnumProperty, IntProperty, StringProperty, CollectionProperty, PointerProperty, BoolProperty
import mathutils
import bmesh
import os
import random

# Generator class to hold name and image array
class Generator(PropertyGroup):
    name: StringProperty(
        name="Name",
        description="Name of this generator",
        default="New Generator"
    )
    
    # We'll use this to store the directory path for images
    image_directory: StringProperty(
        name="Image Directory",
        description="Directory containing images for this generator",
        default="",
        subtype='DIR_PATH'
    )
    
    # Target object in the scene
    target_object: PointerProperty(
        name="Target",
        description="Target object to use with this generator",
        type=bpy.types.Object
    )
    
    # Mask texture name
    mask_texture: StringProperty(
        name="Mask Texture",
        description="Name of the mask texture for this generator",
        default=""
    )

# Operator to add a new generator
class AddGenerator(Operator):
    bl_idname = "object.add_generator"
    bl_label = "Add Generator"
    bl_options = {'REGISTER', 'UNDO'}
    
    def execute(self, context):
        # Add a new generator to the collection
        generator = context.scene.generators.add()
        generator.name = f"Generator {len(context.scene.generators)}"
        return {'FINISHED'}

class ToggleMask(Operator):
    bl_idname = "object.toggle_mask"
    bl_label = "Toggle Mask"
    bl_options = {'REGISTER', 'UNDO'}
    
    index: IntProperty(
        name="Index",
        description="Index of the generator to toggle mask for",
        default=0,
        min=0
    )
    
    def execute(self, context):
        if self.index >= len(context.scene.generators):
            self.report({'ERROR'}, "Invalid generator index")
            return {'CANCELLED'}
            
        generator = context.scene.generators[self.index]
        
        # Check if target and mask exist
        if not generator.target_object:
            self.report({'ERROR'}, "No target object selected")
            return {'CANCELLED'}
            
        if not generator.mask_texture or generator.mask_texture not in bpy.data.images:
            self.report({'ERROR'}, "Mask texture not found. Create a mask first.")
            return {'CANCELLED'}
            
        target_obj = generator.target_object
        mask_img = bpy.data.images[generator.mask_texture]
        
        # Toggle the visibility
        current_state = context.scene.show_mask
        context.scene.show_mask = not current_state
        
        # Get material
        if not target_obj.data.materials or len(target_obj.data.materials) == 0:
            self.report({'ERROR'}, "Target object has no materials")
            return {'CANCELLED'}
            
        mat = target_obj.data.materials[0]
        if not mat.use_nodes:
            self.report({'ERROR'}, "Material does not use nodes")
            return {'CANCELLED'}
            
        nodes = mat.node_tree.nodes
        links = mat.node_tree.links
        
        # Find the mask node
        mask_node = None
        for node in nodes:
            if node.type == 'TEX_IMAGE' and node.name == 'Mask Texture':
                mask_node = node
                break
                
        # If mask node not found, we can't toggle it
        if not mask_node:
            self.report({'ERROR'}, "Mask texture node not found in material")
            return {'CANCELLED'}
            
        # Find a Principled BSDF or similar node
        shader_node = None
        for node in nodes:
            if node.type == 'BSDF_PRINCIPLED':
                shader_node = node
                break
            elif node.type in ['BSDF_DIFFUSE', 'BSDF_GLOSSY']:
                shader_node = node
                break
                
        if not shader_node:
            self.report({'ERROR'}, "No shader node found in material")
            return {'CANCELLED'}
        
        if not context.scene.show_mask:
            # Hide mask - store the current node setup
            # First check if we have an original texture stored as a custom property
            if "original_texture" in target_obj:
                # Get the texture name from the custom property
                tex_name = target_obj["original_texture"]
                if tex_name in bpy.data.images:
                    # Create a texture node for the original texture
                    texture_node = nodes.new('ShaderNodeTexImage')
                    texture_node.name = 'Original Texture'
                    texture_node.image = bpy.data.images[tex_name]
                    texture_node.location = (-300, 300)
                    
                    # Disconnect mask from Base Color
                    for link in list(links):
                        if (link.from_node == mask_node and 
                            link.to_node == shader_node and
                            link.to_socket.name == 'Base Color'):
                            links.remove(link)
                    
                    # Connect original texture to Base Color
                    links.new(texture_node.outputs['Color'], shader_node.inputs['Base Color'])
                    
                    self.report({'INFO'}, f"Restored original texture: {tex_name}")
                else:
                    self.report({'WARNING'}, f"Original texture {tex_name} not found")
            else:
                # Just disconnect the mask
                for link in list(links):
                    if (link.from_node == mask_node and 
                        link.to_node == shader_node and
                        link.to_socket.name == 'Base Color'):
                        links.remove(link)
                
                self.report({'INFO'}, "Mask hidden but no original texture found")
        else:
            # Show mask - first save current texture if connected
            color_input = shader_node.inputs['Base Color']
            
            # Store the current texture if it's linked
            if color_input.is_linked:
                color_source = color_input.links[0].from_node
                if color_source.type == 'TEX_IMAGE' and color_source.image:
                    # Save the current texture name as a custom property
                    target_obj["original_texture"] = color_source.image.name
                    self.report({'INFO'}, f"Saved original texture: {color_source.image.name}")
                
                # Disconnect current texture
                for link in list(color_input.links):
                    links.remove(link)
            
            # Connect mask to Base Color
            links.new(mask_node.outputs['Color'], shader_node.inputs['Base Color'])
            self.report({'INFO'}, "Mask shown")
        
        return {'FINISHED'}
    
    

# Operator to remove a generator
class RemoveGenerator(Operator):
    bl_idname = "object.remove_generator"
    bl_label = "Remove Generator"
    bl_options = {'REGISTER', 'UNDO'}
    
    index: IntProperty(
        name="Index",
        description="Index of the generator to remove",
        default=0,
        min=0
    )
    
    def execute(self, context):
        # Remove the generator at the specified index
        if self.index < len(context.scene.generators):
            context.scene.generators.remove(self.index)
            # Update the active generator index if needed
            if context.scene.active_generator_index >= len(context.scene.generators):
                context.scene.active_generator_index = max(0, len(context.scene.generators) - 1)
        return {'FINISHED'}

# Operator to browse for image directory
class BrowseImageDirectory(Operator):
    bl_idname = "object.browse_image_directory"
    bl_label = "Browse Images"
    bl_options = {'REGISTER', 'UNDO'}
    
    index: IntProperty(
        name="Index",
        description="Index of the generator to update",
        default=0,
        min=0
    )
    
    def execute(self, context):
        # This doesn't actually browse, just updates the UI
        # The actual browsing is done via the UI's file browser
        return {'FINISHED'}
        
# Operator to create a mask texture
class CreateMask(Operator):
    bl_idname = "object.create_mask"
    bl_label = "Create Mask"
    bl_options = {'REGISTER', 'UNDO'}
    
    index: IntProperty(
        name="Index",
        description="Index of the generator to create mask for",
        default=0,
        min=0
    )
    
    def execute(self, context):
        if self.index >= len(context.scene.generators):
            self.report({'ERROR'}, "Invalid generator index")
            return {'CANCELLED'}
            
        generator = context.scene.generators[self.index]
        
        # Make sure a target is selected
        if not generator.target_object:
            self.report({'ERROR'}, "You must select a target object first")
            return {'CANCELLED'}
            
        target_obj = generator.target_object
        
        # Create a new mask texture with a visible checkerboard pattern for testing
        texture_name = f"Mask_{generator.name}"
        
        # Check if the texture already exists
        if texture_name in bpy.data.images:
            mask_texture = bpy.data.images[texture_name]
        else:
            # Create a new image
            mask_texture = bpy.data.images.new(
                name=texture_name,
                width=1024,
                height=1024,
                alpha=True
            )
            
            # Create a checkerboard pattern for testing
            pixels = []
            for y in range(1024):
                for x in range(1024):
                    # Create a large checkerboard pattern (8x8 squares)
                    checker_x = (x // 128) % 2
                    checker_y = (y // 128) % 2
                    
                    if (checker_x == 0 and checker_y == 0) or (checker_x == 1 and checker_y == 1):
                        pixels.extend([1.0, 1.0, 1.0, 1.0])  # White square
                    else:
                        pixels.extend([0.0, 0.0, 0.0, 1.0])  # Black square
            
            mask_texture.pixels = pixels
            mask_texture.update()
            
        # Store the texture name in the generator
        generator.mask_texture = texture_name
        
        # Make sure target object has a material
        if not target_obj.data.materials:
            mat = bpy.data.materials.new(name=f"MaskMaterial_{generator.name}")
            mat.use_nodes = True
            target_obj.data.materials.append(mat)
        else:
            mat = target_obj.data.materials[0]
            if not mat.use_nodes:
                mat.use_nodes = True
        
        # Set up nodes for the mask texture
        nodes = mat.node_tree.nodes
        links = mat.node_tree.links
        
        # Clear existing nodes
        for node in nodes:
            nodes.remove(node)
        
        # Add principled BSDF node
        bsdf_node = nodes.new('ShaderNodeBsdfPrincipled')
        bsdf_node.location = (0, 0)
        
        # Add output node
        output_node = nodes.new('ShaderNodeOutputMaterial')
        output_node.location = (300, 0)
        
        # Add mask texture node with explicit UV mapping
        mask_node = nodes.new('ShaderNodeTexImage')
        mask_node.name = 'Mask Texture'
        mask_node.location = (-300, 0)
        mask_node.image = mask_texture
        
        # Add UV map node explicitly
        uv_node = nodes.new('ShaderNodeUVMap')
        uv_node.location = (-500, -100)
        
        # Connect nodes
        links.new(uv_node.outputs['UV'], mask_node.inputs['Vector'])
        links.new(mask_node.outputs['Color'], bsdf_node.inputs['Base Color'])
        links.new(bsdf_node.outputs['BSDF'], output_node.inputs['Surface'])
        
        # Open the image editor with the mask
        self.open_image_editor(context, mask_texture)
        
        self.report({'INFO'}, f"Created mask texture: {texture_name} with checkerboard pattern")
        self.report({'INFO'}, f"You can now edit the mask in the Image Editor")
        
        return {'FINISHED'}
    
    def open_image_editor(self, context, image):
        # Try to find an image editor area
        image_editor = None
        for area in context.screen.areas:
            if area.type == 'IMAGE_EDITOR':
                image_editor = area
                break
        
        # If found, set the image
        if image_editor:
            image_editor.spaces.active.image = image

# Operator to place selected images from a generator
class PlaceImages(Operator):
    bl_idname = "object.place_images"
    bl_label = "Place Images"
    bl_options = {'REGISTER', 'UNDO'}
    
    index: IntProperty(
        name="Index",
        description="Index of the generator to use",
        default=0,
        min=0
    )
    
    use_mask: BoolProperty(
        name="Use Mask",
        description="Place images only in masked areas",
        default=True
    )
    
    placement_density: FloatProperty(
        name="Density",
        description="Density of image placement (images per square meter)",
        default=5.0,
        min=0.1,
        max=50.0
    )
    
    padding: FloatProperty(
        name="Padding",
        description="Extra padding around the cropped area (in percent)",
        default=0.0,
        min=0.0,
        max=100.0
    )
    
    def execute(self, context):
        if self.index >= len(context.scene.generators):
            self.report({'ERROR'}, "Invalid generator index")
            return {'CANCELLED'}
            
        generator = context.scene.generators[self.index]
        
        # Clear previous generated objects if this is a regeneration
        self.clear_previous_generated(context, generator)
        
        # Check if a target object is selected
        if not generator.target_object:
            self.report({'ERROR'}, "No target object selected")
            return {'CANCELLED'}
            
        target_obj = generator.target_object
        
        # Check if mask exists and is required
        if self.use_mask and (not generator.mask_texture or generator.mask_texture not in bpy.data.images):
            self.report({'ERROR'}, "Mask texture not found. Create a mask first.")
            return {'CANCELLED'}
            
        # Get the directory path
        directory = generator.image_directory
        if not directory or not os.path.exists(directory):
            self.report({'ERROR'}, f"Directory doesn't exist: {directory}")
            return {'CANCELLED'}
            
        # Get image files from the directory
        image_extensions = {'.png', '.jpg', '.jpeg', '.tiff', '.bmp', '.tga'}
        image_files = []
        
        try:
            for filename in os.listdir(directory):
                if any(filename.lower().endswith(ext) for ext in image_extensions):
                    image_files.append(filename)
        except Exception as e:
            self.report({'ERROR'}, f"Error reading directory: {str(e)}")
            return {'CANCELLED'}
            
        if not image_files:
            self.report({'WARNING'}, "No image files found in directory")
            return {'CANCELLED'}
        
        # Load all images first
        images = []
        for img_file in image_files:
            img_path = os.path.join(directory, img_file)
            try:
                img = bpy.data.images.load(img_path)
                images.append(img)
            except Exception as e:
                self.report({'WARNING'}, f"Couldn't load {img_file}: {str(e)}")
        
        if not images:
            self.report({'ERROR'}, "No images could be loaded")
            return {'CANCELLED'}

        # Get mask image if we're using it
        mask_img = None
        if self.use_mask:
            mask_img = bpy.data.images[generator.mask_texture]
            mask_img.update()
            mask_width, mask_height = mask_img.size
            pixels = np.array(mask_img.pixels[:]).reshape(mask_height, mask_width, 4)
            
            # Report stats about mask
            white_count = np.sum(pixels[:,:,0] > 0.5)
            black_count = np.sum(pixels[:,:,0] < 0.1)
            total_pixels = mask_width * mask_height
            self.report({'INFO'}, f"Mask has {white_count} white pixels and {black_count} black pixels out of {total_pixels}")
        
        # Get target dimensions
        target_dim_x = target_obj.dimensions.x
        target_dim_y = target_obj.dimensions.y
        target_area = target_dim_x * target_dim_y
        
        # Calculate grid size based on density
        num_points = int(target_area * self.placement_density * 4)
        grid_size = max(int(math.sqrt(num_points)), 20)
        
        self.report({'INFO'}, f"Target dimensions: {target_dim_x:.2f}m x {target_dim_y:.2f}m, Area: {target_area:.2f}m²")
        self.report({'INFO'}, f"Using grid size {grid_size}x{grid_size} to place ~{int(target_area * self.placement_density)} images")
        
        # Get target object world matrix and bounds
        world_matrix = target_obj.matrix_world
        local_min = target_obj.bound_box[0]
        local_max = target_obj.bound_box[6]
        
        # Create points directly on the target object
        points = []
        
        # Use simple grid placement on object surface
        for i in range(grid_size):
            for j in range(grid_size):
                # Calculate position in object's local space
                u = i / (grid_size - 1)  # 0 to 1
                v = j / (grid_size - 1)  # 0 to 1
                
                # Get local coordinates
                local_x = local_min[0] + u * (local_max[0] - local_min[0])
                local_y = local_min[1] + v * (local_max[1] - local_min[1])
                local_z = local_min[2]
                
                # Check mask if needed
                if self.use_mask:
                    # Sample mask at this UV coordinate
                    px = int(u * (mask_width - 1))
                    py = int(v * (mask_height - 1))
                    
                    # Get pixel value (use red channel)
                    pixel_value = pixels[py, px, 0]
                    
                    # Skip if pixel is black (below threshold)
                    if pixel_value < 0.2:
                        continue
                
                # Convert to world space
                world_pos = world_matrix @ mathutils.Vector((local_x, local_y, local_z))
                
                # Add small random offset for natural look
                offset_scale = min(target_dim_x, target_dim_y) / grid_size / 2
                offset_x = random.uniform(-offset_scale, offset_scale)
                offset_y = random.uniform(-offset_scale, offset_scale)  # Fixed: was using offset_y
                
                points.append((
                    world_pos.x + offset_x,
                    world_pos.y + offset_y,
                    world_pos.z
                ))
        
        # Now limit to the actual desired density if we found more points than needed
        target_count = int(target_area * self.placement_density)
        if len(points) > target_count:
            # Randomly select subset of points to match desired density
            points = random.sample(points, target_count)
        
        # Place objects at the valid points
        if not points:
            self.report({'WARNING'}, "No valid placement points found. Check your mask or increase density.")
            return {'CANCELLED'}
            
        self.report({'INFO'}, f"Placing {len(points)} objects on {target_obj.name}")
        
        # Store the original selection and active object
        original_selection = context.selected_objects.copy()
        original_active = context.active_object
        original_cursor = context.scene.cursor.location.copy()
        
        # Store important scene settings
        rotation_angle = context.scene.rotation_angle
        rotation_axis = context.scene.rotation_axis
        stretch_axis = context.scene.stretch_axis
        pixels_per_meter = context.scene.pixels_per_meter
        
        # Create a collection to hold all generated objects for this generator
        collection_name = f"Generated_{generator.name}"
        if collection_name in bpy.data.collections:
            generated_collection = bpy.data.collections[collection_name]
        else:
            generated_collection = bpy.data.collections.new(collection_name)
            context.scene.collection.children.link(generated_collection)
        
        # Dictionary to store our image prototypes
        image_prototypes = {}
        
        # For each placement point
        for point in points:
            # Pick a random image
            img = random.choice(images)
            img_name = img.name
            
            # Check if we already have a prototype for this image
            if img_name not in image_prototypes:
                # Create the prototype
                prototype = self.create_image_prototype(context, img, 
                                        padding=self.padding,
                                        pixels_per_meter=pixels_per_meter,
                                        rotation_angle=rotation_angle,
                                        rotation_axis=rotation_axis,
                                        stretch_axis=stretch_axis)
                
                # Move to our collection and hide from view
                if prototype.name in context.scene.collection.objects:
                    context.scene.collection.objects.unlink(prototype)
                generated_collection.objects.link(prototype)
                prototype.hide_viewport = True
                prototype.hide_render = True
                
                # Store it
                image_prototypes[img_name] = prototype
            else:
                # Use existing prototype
                prototype = image_prototypes[img_name]
            
            # Create instance at the placement point
            instance = self.create_instance_at_point(context, prototype, point)
            
            # Move to our collection
            if instance:
                # Make sure instance is in scene collection before unlinking
                if instance.name in context.scene.collection.objects:
                    context.scene.collection.objects.unlink(instance)
                generated_collection.objects.link(instance)
                
                # Tag as a generated object for this generator
                instance["generated_by"] = generator.name
        
        # Clean up - tag prototype objects so we can find them later
        for prototype in image_prototypes.values():
            prototype["is_prototype"] = True
            prototype["generated_by"] = generator.name
        
        # Restore original selection
        bpy.ops.object.select_all(action='DESELECT')
        for obj in original_selection:
            if obj:  # Ensure object still exists
                obj.select_set(True)
        if original_active:
            context.view_layer.objects.active = original_active
            
        # Restore cursor location
        context.scene.cursor.location = original_cursor
            
        return {'FINISHED'}
    
    def clear_previous_generated(self, context, generator):
        """Clear previously generated objects for this generator"""
        
        # First check if there's a collection for this generator
        collection_name = f"Generated_{generator.name}"
        if collection_name in bpy.data.collections:
            # Get the collection
            collection = bpy.data.collections[collection_name]
            
            # Remove all objects in the collection
            for obj in list(collection.objects):  # Use list to avoid issues with changing collection
                # Unlink from all collections first
                for coll in obj.users_collection:
                    coll.objects.unlink(obj)
                    
                # Delete the object data
                if obj.data:
                    if hasattr(obj.data, 'users') and obj.data.users == 1:
                        if hasattr(bpy.data, obj.data.bl_rna.identifier):
                            data_collection = getattr(bpy.data, obj.data.bl_rna.identifier)
                            data_collection.remove(obj.data)
                
                # Delete the object
                bpy.data.objects.remove(obj)
            
            # Now remove the collection itself
            bpy.data.collections.remove(collection)
            
        # Just to be thorough, also check for any stray objects
        objects_to_remove = []
        for obj in bpy.data.objects:
            if "generated_by" in obj and obj["generated_by"] == generator.name:
                objects_to_remove.append(obj)
        
        # Remove any stray objects
        for obj in objects_to_remove:
            bpy.data.objects.remove(obj)
        
        # Also check for empty collections
        for coll in list(bpy.data.collections):
            if coll.name.startswith(f"Generated_{generator.name}") and len(coll.objects) == 0:
                bpy.data.collections.remove(coll)
        
        self.report({'INFO'}, f"Cleared previous generated objects for {generator.name}")
    
    # Corrected material handling for instancing:

    def create_instance_at_point(self, context, prototype, location):
        """Create an instance of the prototype at the specified location"""
        if not prototype:
            return None
            
        # Create the instance - using the same mesh data as the prototype
        instance = bpy.data.objects.new(f"{prototype.name}_instance", prototype.data)
        
        # Set the instance's location
        instance.location = location
        
        # Handle materials - instances need materials explicitly assigned
        if prototype.data.materials:
            # Make sure the instance has materials slots
            while len(instance.material_slots) < len(prototype.data.materials):
                bpy.ops.object.material_slot_add({'object': instance})
                
            # Assign the same materials from the prototype
            for i, mat in enumerate(prototype.data.materials):
                if i < len(instance.material_slots):
                    instance.material_slots[i].material = mat
        
        # Copy all transformations except location
        instance.rotation_euler = prototype.rotation_euler.copy()
        instance.scale = prototype.scale.copy()
        
        # Link to scene
        context.scene.collection.objects.link(instance)
        
        return instance
    
    def create_image_prototype(self, context, img, padding=0.0, 
                             pixels_per_meter=32.0, rotation_angle=45.0, 
                             rotation_axis='X', stretch_axis='Y'):
        """Create a prototype object that will be instanced - following the correct order:
        spawn -> crop -> rotate/scale -> origin"""
        
        # 1. SPAWN: Create basic plane at origin for initial processing
        bpy.ops.mesh.primitive_plane_add(size=1, location=(0, 0, 0))
        plane = context.active_object
        plane.name = f"Proto_{os.path.splitext(img.name)[0]}"
        
        # Apply material with image
        mat = bpy.data.materials.new(name=f"Mat_{img.name}")
        mat.use_nodes = True
        nodes = mat.node_tree.nodes
        links = mat.node_tree.links
        
        # Clear existing nodes
        for node in nodes:
            nodes.remove(node)
        
        # Add texture node
        texture_node = nodes.new('ShaderNodeTexImage')
        texture_node.name = 'Image Texture'
        texture_node.image = img
        texture_node.location = (-300, 300)
        
        # Add principled BSDF node
        bsdf_node = nodes.new('ShaderNodeBsdfPrincipled')
        bsdf_node.location = (0, 300)
        
        # Add output node
        output_node = nodes.new('ShaderNodeOutputMaterial')
        output_node.location = (300, 300)
        
        # Connect nodes
        links.new(texture_node.outputs['Color'], bsdf_node.inputs['Base Color'])
        links.new(texture_node.outputs['Alpha'], bsdf_node.inputs['Alpha'])
        links.new(bsdf_node.outputs['BSDF'], output_node.inputs['Surface'])
        
        # Set material settings
        mat.blend_method = 'HASHED'
        
        # Apply the material
        plane.data.materials.clear()
        plane.data.materials.append(mat)
        
        # 2. CROP: Crop to alpha bounds
        self.crop_plane_to_alpha(context, plane, padding)
        
        # 3. ROTATE AND SCALE:
        # Scale based on pixels per meter
        width, height = img.size
        aspect_ratio = width / height
        scale_factor = height / pixels_per_meter
        
        plane.scale.x = scale_factor * aspect_ratio
        plane.scale.y = scale_factor
        
        # Apply rotation based on global settings
        angle_rad = math.radians(rotation_angle)
        if rotation_axis == 'X':
            plane.rotation_euler.x = angle_rad
        elif rotation_axis == 'Y':
            plane.rotation_euler.y = angle_rad
        elif rotation_axis == 'Z':
            plane.rotation_euler.z = angle_rad
            
        # Apply the scaling
        bpy.ops.object.transform_apply(location=False, rotation=False, scale=True)
        
        # Adjust for stretch axis
        if rotation_axis == 'X' and stretch_axis == 'Y':
            # Calculate the stretch factor
            cos_angle = math.cos(angle_rad)
            stretch_factor = 1 / cos_angle if cos_angle != 0 else 1.0
            
            # Apply stretch
            if stretch_axis == 'X':
                plane.scale.x = stretch_factor
            elif stretch_axis == 'Y':
                plane.scale.y = stretch_factor
            else:  # Z axis
                plane.scale.z = stretch_factor
                
            # Apply the stretching
            bpy.ops.object.transform_apply(location=False, rotation=False, scale=True)
        
        return plane
    
    def crop_plane_to_alpha(self, context, plane, padding=0.0):
        """Crop a plane to its image's alpha bounds"""
        
        # Get the image from the plane's material
        if not plane.data.materials or not plane.data.materials[0] or not plane.data.materials[0].node_tree:
            return
            
        material = plane.data.materials[0]
        nodes = material.node_tree.nodes
        image_node = nodes.get('Image Texture')
        
        if not image_node or not image_node.image:
            return
            
        image = image_node.image
        mesh = plane.data
            
        # Get image bounds
        bounds = self.get_image_bounds(image)
        if not bounds:
            return
            
        # Apply padding
        padding_factor = padding / 100
        bounds['min_u'] = max(0.0, bounds['min_u'] - padding_factor)
        bounds['max_u'] = min(1.0, bounds['max_u'] + padding_factor)
        bounds['min_v'] = max(0.0, bounds['min_v'] - padding_factor)
        bounds['max_v'] = min(1.0, bounds['max_v'] + padding_factor)
        
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
        
        # Set origin to bottom center
        original_location = plane.location.copy()
        
        # Calculate bounds in local space after cropping
        x_coords = [v.co.x for v in mesh.vertices]
        y_coords = [v.co.y for v in mesh.vertices]
        z_coords = [v.co.z for v in mesh.vertices]
        min_x, max_x = min(x_coords), max(x_coords)
        min_y, max_y = min(y_coords), max(y_coords)
        min_z, max_z = min(z_coords), max(z_coords)
        center_x = (min_x + max_x) / 2
        center_y = (min_y + max_y) / 2
        
        # Set origin to bottom center
        orig_cursor_loc = context.scene.cursor.location.copy()
        
        # Position cursor at desired origin in object's local space
        context.scene.cursor.location = plane.matrix_world @ mathutils.Vector((center_x, min_y, 0))
        
        # Set origin to cursor
        bpy.ops.object.origin_set(type='ORIGIN_CURSOR')
        
        # Restore cursor
        context.scene.cursor.location = orig_cursor_loc
    
    def get_image_bounds(self, image):
        """Calculate the bounds of an image based on alpha channel"""
        
        if not image or not image.pixels:
            return None
            
        # Convert image pixels to numpy array
        width = image.size[0]
        height = image.size[1]
        pixels = np.array(image.pixels[:]).reshape(height, width, 4)
        
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
        
        # Convert to UV coordinates
        return {
            'min_u': min_col / width,
            'max_u': (max_col + 1) / width,
            'min_v': min_row / height,
            'max_v': (max_row + 1) / height
        }
            
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
            Bounds = CropImages.GetImageBounds(image_node.image)
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

#Adjust height based on pixels per meter and angle
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

# Panel for displaying generators
class GeneratorListPanel(Panel):
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'UI'
    bl_category = 'Plane tools'
    bl_label = "Procedural Placement"
    bl_options = {'DEFAULT_CLOSED'}
    
    # This is how your draw method in the GeneratorListPanel class should look

    # Fix for the None error in the panel draw method
    def draw(self, context):
        layout = self.layout
        scene = context.scene
        
        # Add button to create a new generator
        row = layout.row()
        row.operator("object.add_generator", text="Add Generator")
        
        # Draw generators
        for i, generator in enumerate(scene.generators):
            box = layout.box()
            row = box.row()
            row.prop(generator, "name", text="")
            remove_op = row.operator("object.remove_generator", text="", icon='X')
            if remove_op:  # Check for None
                remove_op.index = i
            
            # Target object selection
            box.prop(generator, "target_object", text="Target")
            
            # Image directory selection
            box.prop(generator, "image_directory", text="Directory")
            
            # Mask creation and display
            row = box.row()
            create_mask_op = row.operator("object.create_mask", text="Create Mask")
            if create_mask_op:  # Check for None
                create_mask_op.index = i
            
            # Add mask toggle button if a mask exists
            if generator.mask_texture and generator.mask_texture in bpy.data.images:
                # Dynamic label based on current state
                show_text = "Hide Mask" if context.scene.show_mask else "Show Mask"
                mask_toggle_op = row.operator("object.toggle_mask", text=show_text)
                if mask_toggle_op:  # Check for None
                    mask_toggle_op.index = i
                
                # Show mask texture name
                row.label(text=generator.mask_texture)
            
            # Place images button with settings
            place_box = box.box()
            place_box.label(text="Place Images:")
            
            row = place_box.row()
            row.prop(context.scene, "placement_density", text="Density")
            row.prop(context.scene, "use_mask_for_placement", text="Use Mask")
            
            place_op = place_box.operator("object.place_images", text="Generate")
            if place_op:  # Check for None
                place_op.index = i
                place_op.use_mask = context.scene.use_mask_for_placement
                place_op.placement_density = context.scene.placement_density

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
        
class ToggleMask(Operator):
    bl_idname = "object.toggle_mask"
    bl_label = "Toggle Mask"
    bl_options = {'REGISTER', 'UNDO'}
    
    index: IntProperty(
        name="Index",
        description="Index of the generator to toggle mask for",
        default=0,
        min=0
    )
    
    def execute(self, context):
        if self.index >= len(context.scene.generators):
            self.report({'ERROR'}, "Invalid generator index")
            return {'CANCELLED'}
            
        generator = context.scene.generators[self.index]
        
        # Check if target and mask exist
        if not generator.target_object:
            self.report({'ERROR'}, "No target object selected")
            return {'CANCELLED'}
            
        if not generator.mask_texture or generator.mask_texture not in bpy.data.images:
            self.report({'ERROR'}, "Mask texture not found. Create a mask first.")
            return {'CANCELLED'}
            
        target_obj = generator.target_object
        mask_img = bpy.data.images[generator.mask_texture]
        
        # Toggle the visibility
        current_state = context.scene.show_mask
        context.scene.show_mask = not current_state
        
        # Get material
        if not target_obj.data.materials or len(target_obj.data.materials) == 0:
            self.report({'ERROR'}, "Target object has no materials")
            return {'CANCELLED'}
            
        mat = target_obj.data.materials[0]
        if not mat.use_nodes:
            self.report({'ERROR'}, "Material does not use nodes")
            return {'CANCELLED'}
            
        nodes = mat.node_tree.nodes
        links = mat.node_tree.links
        
        # Find the mask node
        mask_node = None
        for node in nodes:
            if node.type == 'TEX_IMAGE' and node.name == 'Mask Texture':
                mask_node = node
                break
                
        # If mask node not found, we can't toggle it
        if not mask_node:
            self.report({'ERROR'}, "Mask texture node not found in material")
            return {'CANCELLED'}
            
        # Find a Principled BSDF or similar node
        shader_node = None
        for node in nodes:
            if node.type == 'BSDF_PRINCIPLED':
                shader_node = node
                break
            elif node.type in ['BSDF_DIFFUSE', 'BSDF_GLOSSY']:
                shader_node = node
                break
                
        if not shader_node:
            self.report({'ERROR'}, "No shader node found in material")
            return {'CANCELLED'}
            
        # Find output node
        output_node = None
        for node in nodes:
            if node.type == 'OUTPUT_MATERIAL':
                output_node = node
                break
                
        if not output_node:
            self.report({'ERROR'}, "No output node found in material")
            return {'CANCELLED'}
        
        # Toggle connections based on new state
        if not context.scene.show_mask:
            # Hide mask - disconnect from Base Color
            for link in list(links):
                if (link.from_node == mask_node and 
                    link.to_node == shader_node and
                    link.to_socket.name == 'Base Color'):
                    links.remove(link)
            
            # Check if there's a stored original Base Color node
            original_color_node = None
            for node in nodes:
                if node.name == 'Original Color':
                    original_color_node = node
                    break
                
            # If found, reconnect it
            if original_color_node:
                links.new(original_color_node.outputs[0], shader_node.inputs['Base Color'])
            
            self.report({'INFO'}, "Mask hidden")
        else:
            # Show mask - first save current Base Color connection
            color_input = shader_node.inputs['Base Color']
            if color_input.is_linked:
                color_link = color_input.links[0]
                color_source = color_link.from_node
                
                # Rename to identify it
                color_source.name = 'Original Color'
                
                # Disconnect it
                links.remove(color_link)
            
            # Connect mask to Base Color
            links.new(mask_node.outputs['Color'], shader_node.inputs['Base Color'])
            self.report({'INFO'}, "Mask shown")
        
        return {'FINISHED'}

#Adds custom properties to Blender's Scene type and registers classes
def register():
    # Register the Generator class first
    bpy.utils.register_class(Generator)
    
    # Add generators collection to Scene
    bpy.types.Scene.generators = CollectionProperty(
        type=Generator,
        name="Generators",
        description="Collection of image generators"
    )
    
    # Add index for active generator
    bpy.types.Scene.active_generator_index = IntProperty(
        name="Active Generator Index",
        default=0
    )
    
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
    
    # Placement density
    bpy.types.Scene.placement_density = FloatProperty(
        name="Density",
        description="Density of image placement (images per square meter)",
        default=5.0,  # Increased default
        min=0.1,
        max=50.0      # Increased maximum
    )
    
    # Use mask for placement
    bpy.types.Scene.use_mask_for_placement = BoolProperty(
        name="Use Mask for Placement", 
        description="Place images only in masked areas",
        default=True
    )
    
    # Add mask visibility toggle
    bpy.types.Scene.show_mask = BoolProperty(
        name="Show Mask",
        description="Toggle mask visibility on target objects",
        default=True
    )
    
    # Register all operators and panels
    bpy.utils.register_class(CropImages)
    bpy.utils.register_class(RotateAndScale)
    bpy.utils.register_class(RenderPlaneToolsPanel)
    bpy.utils.register_class(AdjustHeight)
    bpy.utils.register_class(CreateStairs)
    
    # Register new procedural placement classes
    bpy.utils.register_class(AddGenerator)
    bpy.utils.register_class(RemoveGenerator)
    bpy.utils.register_class(BrowseImageDirectory)
    bpy.utils.register_class(CreateMask)
    bpy.utils.register_class(ToggleMask)  # Register the new ToggleMask operator
    bpy.utils.register_class(PlaceImages)
    bpy.utils.register_class(GeneratorListPanel)

#Removes custom properties and unregisters classes when disabling the add-on
def unregister():
    # Unregister procedural placement classes
    bpy.utils.unregister_class(GeneratorListPanel)
    bpy.utils.unregister_class(PlaceImages)
    bpy.utils.unregister_class(ToggleMask)  # Unregister the ToggleMask operator
    bpy.utils.unregister_class(CreateMask)
    bpy.utils.unregister_class(BrowseImageDirectory)
    bpy.utils.unregister_class(RemoveGenerator)
    bpy.utils.unregister_class(AddGenerator)
    
    # Unregister original classes
    bpy.utils.unregister_class(RenderPlaneToolsPanel)
    bpy.utils.unregister_class(RotateAndScale)
    bpy.utils.unregister_class(CropImages)
    bpy.utils.unregister_class(AdjustHeight)
    bpy.utils.unregister_class(CreateStairs)
    
    # Unregister Generator class last
    bpy.utils.unregister_class(Generator)
    
    # Delete properties
    del bpy.types.Scene.generators
    del bpy.types.Scene.active_generator_index
    del bpy.types.Scene.rotation_angle
    del bpy.types.Scene.rotation_axis
    del bpy.types.Scene.stretch_axis
    del bpy.types.Scene.pixels_per_meter
    del bpy.types.Scene.num_steps
    del bpy.types.Scene.placement_density
    del bpy.types.Scene.use_mask_for_placement
    del bpy.types.Scene.show_mask  # Delete the show_mask property

#Setup main 
if __name__ == "__main__":
    try:
        unregister()
    except:
        pass
    register()