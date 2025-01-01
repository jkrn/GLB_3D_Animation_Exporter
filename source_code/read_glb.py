import sys
import pygltflib
import numpy as np
import json
import math
import ctypes
from scipy.spatial.transform import Rotation as Rot

# FPS in Blender
fps_blender = 24

glb_filename = ''
binmesh_filename = ''
anim_filename = ''
skel_filename = ''
json_filename = ''
txt_filename = ''

for argument in sys.argv:
    if argument[0:5] == '-glb=':
        glb_filename = argument[5:]
    elif argument[0:9] == '-binmesh=':
        binmesh_filename = argument[9:]
    elif argument[0:6] == '-anim=':
        anim_filename = argument[6:]
    elif argument[0:6] == '-skel=':
        skel_filename = argument[6:]
    elif argument[0:6] == '-json=':
        json_filename = argument[6:]
    elif argument[0:5] == '-txt=':
        txt_filename = argument[5:]

print('glb_filename = '+glb_filename)
print('binmesh_filename = '+binmesh_filename)
print('anim_filename = '+anim_filename)
print('skel_filename = '+skel_filename)
print('json_filename = '+json_filename)
print('txt_filename = '+txt_filename)

glb_file_set = len(glb_filename)>0
binmesh_file_set = len(binmesh_filename)>0
anim_file_set = len(anim_filename)>0
skel_file_set = len(skel_filename)>0
json_file_set = len(json_filename)>0
txt_file_set = len(txt_filename)>0

if glb_file_set == False:
    print('Missing parameter')
    print('USAGE: python read_glb.py -glb=[GLB FILE] OPTIONAL: -binmesh=[BINMESH FILE] -anim=[ANIM FILE] -skel=[SKEL FILE] -json=[JSON FILE] -txt=[TXT FILE]')
    exit()

class JointNode:
    def __init__(self, joint_id):
        self.joint_id = joint_id
        self.name = ''
        self.parent_id = -1
        self.buffer_pos_id = -1

class AnimationChannel:
    def __init__(self, numFrames):
        self.numFrames = numFrames
        self.complete_frames_time = np.arange(1,numFrames+1,1,dtype=np.uint32)
        self.complete_frames_trans = np.zeros((3,numFrames),dtype=np.float32)
        self.complete_frames_scale = np.zeros((3,numFrames),dtype=np.float32)
        self.complete_frames_rot = np.zeros((4,numFrames),dtype=np.float32)

def isValueIsInArray(array,value):
    index = np.where(array == value)[0]
    return len(index)>0

def getInterpolationValue(currentTime,previousTime,nextTime):
    interpolationValue = (currentTime - previousTime) / (nextTime - previousTime)
    return interpolationValue

def lerpPoint(previousPoint, nextPoint, interpolationValue):
     return previousPoint + interpolationValue * (nextPoint - previousPoint)
     
def slerpQuat(previousQuat, nextQuat, interpolationValue):
        dotProduct = np.dot(previousQuat, nextQuat)
        #make sure we take the shortest path in case dot Product is negative
        if(dotProduct < 0.0):
            nextQuat = -nextQuat
            dotProduct = -dotProduct
        #if the two quaternions are too close to each other, just linear interpolate between the 4D vector
        if(dotProduct > 0.9995):
            tempQuat = previousQuat + interpolationValue*(nextQuat - previousQuat)
            tempQuat = tempQuat/np.linalg.norm(tempQuat)
            return tempQuat
        #perform the spherical linear interpolation
        theta_0 = acos(dotProduct)
        theta = interpolationValue * theta_0
        sin_theta = sin(theta)
        sin_theta_0 = sin(theta_0)
        scalePreviousQuat = cos(theta) - dotproduct * sin_theta / sin_theta_0
        scaleNextQuat = sin_theta / sin_theta_0
        return scalePreviousQuat * previousQuat + scaleNextQuat * nextQuat

# READ BINMESH
def read_binmesh(glb,mesh_index,name_binmesh):
    for prim in glb.meshes[mesh_index].primitives:
        
        # Position
        accessor_vert = glb.accessors[prim.attributes.POSITION]
        num_vertices = accessor_vert.count
        type_vertices = accessor_vert.type
        print('NUM POSITIONS: '+str(num_vertices))
        print('TYPE POSITIONS: '+str(type_vertices))
        bv_vert = glb.bufferViews[accessor_vert.bufferView]
        data_vert = glb._glb_data[bv_vert.byteOffset : bv_vert.byteOffset + bv_vert.byteLength]
        positions = np.frombuffer(data_vert, dtype=np.float32)
        positions = np.reshape(positions, (-1, 3))
        print('VERTEX POSITIONS:')
        
        # Weights
        accessor_weights = glb.accessors[prim.attributes.WEIGHTS_0]
        num_weights = accessor_weights.count
        type_weights = accessor_weights.type
        print('NUM WEIGHTS: '+str(num_weights))
        print('TYPE WEIGHTS: '+str(type_weights))
        bv_weights = glb.bufferViews[accessor_weights.bufferView]
        data_weights = glb._glb_data[bv_weights.byteOffset : bv_weights.byteOffset + bv_weights.byteLength]
        weights = np.frombuffer(data_weights, dtype=np.float32)
        weights = np.reshape(weights, (-1, 4))
        print('VERTEX WEIGHTS:')
        
        # Joints
        accessor_joints = glb.accessors[prim.attributes.JOINTS_0]
        num_joints = accessor_joints.count
        type_joints = accessor_joints.type
        print('NUM JOINTS: '+str(num_joints))
        print('TYPE JOINTS: '+str(type_joints))
        bv_joints = glb.bufferViews[accessor_joints.bufferView]
        data_joints = glb._glb_data[bv_joints.byteOffset : bv_joints.byteOffset + bv_joints.byteLength]
        joints = np.frombuffer(data_joints, dtype=np.ubyte)
        joints = np.reshape(joints, (-1, 4))
        print('VERTEX JOINTS:')
        
        # Indices
        accessor = glb.accessors[prim.indices]
        nindices = accessor.count
        bv = glb.bufferViews[accessor.bufferView]
        data = glb._glb_data[bv.byteOffset : bv.byteOffset + bv.byteLength]
        triangles = np.frombuffer(data, dtype=np.uint16)
        triangles = np.reshape(triangles, (-1, 3))
        print('NUM INDICES: '+str(nindices))
        print('NUM TRIANGLES: '+str(int(nindices / 3)))

        # Save in BINMESH file
        file_out = open(name_binmesh, "wb")
        
        # 1) numIndices
        file_out.write(ctypes.c_uint32(nindices))
        
        # 2) numVertices
        file_out.write(ctypes.c_uint32(num_vertices))
        
        # 3) Indices Stream
        for i in range(triangles.shape[0]):
            file_out.write(ctypes.c_uint32(triangles[i,0]))
            file_out.write(ctypes.c_uint32(triangles[i,2]))
            file_out.write(ctypes.c_uint32(triangles[i,1]))

        # 4) Vertices Stream
        for i in range( num_vertices ):
            # POSITION
            file_out.write(ctypes.c_float(positions[i,0]))
            file_out.write(ctypes.c_float(positions[i,1]))
            file_out.write(ctypes.c_float(positions[i,2]))
            # NORMAL
            file_out.write(ctypes.c_float(0))
            file_out.write(ctypes.c_float(0))
            file_out.write(ctypes.c_float(0))
            # TANGENT
            file_out.write(ctypes.c_float(0))
            file_out.write(ctypes.c_float(0))
            file_out.write(ctypes.c_float(0))
            # BINORMAL
            file_out.write(ctypes.c_float(0))
            file_out.write(ctypes.c_float(0))
            file_out.write(ctypes.c_float(0))
            # COLOR
            file_out.write(ctypes.c_float(1))
            file_out.write(ctypes.c_float(1))
            file_out.write(ctypes.c_float(1))
            file_out.write(ctypes.c_float(1))
            # TEXCOORD
            file_out.write(ctypes.c_float(0))
            file_out.write(ctypes.c_float(0))
            # WEIGHT
            file_out.write(ctypes.c_float(weights[i,0]))
            file_out.write(ctypes.c_float(weights[i,1]))
            file_out.write(ctypes.c_float(weights[i,2]))
            file_out.write(ctypes.c_float(weights[i,3]))
            # JOINT
            file_out.write(ctypes.c_uint32(joints[i,0]))
            file_out.write(ctypes.c_uint32(joints[i,1]))
            file_out.write(ctypes.c_uint32(joints[i,2]))
            file_out.write(ctypes.c_uint32(joints[i,3]))
            
        file_out.close()
        

# READ SKEL AND ANIM
def read_skel_anim(glb,anim_index,time_mult,read_skel,name_skel,read_anim,name_anim,export_txt,name_txt):
    
    # Get joint nodes
    node_list = []
    for curr_joint_index in range(len(glb.skins[0].joints)):
        node_list.append(glb.skins[0].joints[curr_joint_index])
    num_nodes = len(node_list)
    print('NUM NODES:'+str(num_nodes))
    print('NODES:'+str(node_list))
    
    # READ SKEL
    if read_skel:
        
        # Node reference list
        node_reference_list = []
        for curr_joint_index in range(len(glb.skins[0].joints)):
            node_reference_list.append(JointNode(curr_joint_index))
        
        # Set buffer pos id
        for curr_buffer_pos_id in range(len(glb.skins[0].joints)):
            curr_buffer_joint_id = glb.skins[0].joints[curr_buffer_pos_id]
            node_reference_list[curr_buffer_joint_id].buffer_pos_id = curr_buffer_pos_id
        
        for curr_joint_index in range(len(node_reference_list)):
            # Set name
            node_reference_list[curr_joint_index].name = glb.nodes[curr_joint_index].name
            # Check if joint has children
            if hasattr(glb.nodes[curr_joint_index], 'children') and len(glb.nodes[curr_joint_index].children) > 0:
                # print children nodes
                # curr_joint_index is now parent node id
                for child_id in glb.nodes[curr_joint_index].children:
                    node_reference_list[child_id].parent_id = curr_joint_index

        # Chain list
        # Init empty chain lists
        chain_list = []
        for curr_node in range(len(node_reference_list)):
            chain_list.append([])

        # Fill chain list
        for curr_node_id in range(len(node_reference_list)):
            curr_joint_node = node_reference_list[curr_node_id]
            curr_chain_list = chain_list[curr_joint_node.buffer_pos_id]
            curr_chain_list.append(curr_joint_node.buffer_pos_id)
            while curr_joint_node.parent_id != -1:
                curr_joint_node = node_reference_list[curr_joint_node.parent_id]
                curr_chain_list.append(curr_joint_node.buffer_pos_id)

        print('CHAIN LIST: '+str(chain_list))
            
        if export_txt:
            # Get skeleton structure
            file_skeleton_out = open(name_txt, "w")
            for curr_node_idx in node_list:
                file_skeleton_out.write('============='+'\n')
                if hasattr(glb.nodes[curr_node_idx], 'name'):
                    file_skeleton_out.write('NODE NAME '+str(curr_node_idx)+': '+glb.nodes[curr_node_idx].name+'\n')
                if hasattr(glb.nodes[curr_node_idx], 'translation'):
                    file_skeleton_out.write('NODE TRANSLATION '+str(curr_node_idx)+': '+str(glb.nodes[curr_node_idx].translation)+'\n')
                if hasattr(glb.nodes[curr_node_idx], 'rotation'):
                    file_skeleton_out.write('NODE ROTATION '+str(curr_node_idx)+': '+str(glb.nodes[curr_node_idx].rotation)+'\n')
                if hasattr(glb.nodes[curr_node_idx], 'children'):
                    file_skeleton_out.write('NODE CHILDEN '+str(curr_node_idx)+': '+str(glb.nodes[curr_node_idx].children)+'\n')
                    if len(glb.nodes[curr_node_idx].children) > 0:
                        file_skeleton_out.write('NODE CHILDEN '+str(curr_node_idx)+': '+glb.nodes[glb.nodes[curr_node_idx].children[0]].name+'\n')
        
        # Get Inverse Bind Matrices
        accessor_inverse_bind_matrices = glb.accessors[glb.skins[0].inverseBindMatrices]
        num_inverse_bind_matrices = accessor_inverse_bind_matrices.count
        type_inverse_bind_matrices = accessor_inverse_bind_matrices.type
        print('NUM INVERSE BIND MATRICES: '+str(num_inverse_bind_matrices))
        print('TYPE INVERSE BIND MATRICES: '+str(type_inverse_bind_matrices))
        bv_inverse_bind_matrices = glb.bufferViews[accessor_inverse_bind_matrices.bufferView]
        data_inverse_bind_matrices = glb._glb_data[bv_inverse_bind_matrices.byteOffset : bv_inverse_bind_matrices.byteOffset + bv_inverse_bind_matrices.byteLength]
        ibm = np.frombuffer(data_inverse_bind_matrices, dtype=np.float32)
        ibm = np.reshape(ibm, (-1, 16))
        
        # Save SKEL file
        file_out = open(name_skel, "wb")
        # 1) NUM JOINTS
        file_out.write(ctypes.c_uint32(num_inverse_bind_matrices))
        # 2) INVERSE BIND MATRICES
        for i in range( num_inverse_bind_matrices ):
            for j in range(16):
                file_out.write( ctypes.c_float( ibm[i,j]) )
        #3) CHAIN LIST
        for curr_list in chain_list:
            for node_id in curr_list:
                file_out.write(ctypes.c_int32(node_id))
            file_out.write(ctypes.c_int32(-1))
        file_out.close()
        
    # READ ANIM
    if read_anim:
        
        #Get animation
        animation = glb.animations[anim_index]

        # Number of animation channels
        num_animation_channels = len(animation.channels)
        print('NUM CHANNELS: '+str(num_animation_channels))
        
        # Get max time
        num_max_frames = 1
        for curr_channel_index in range(num_animation_channels):
            curr_sampler_index = animation.channels[curr_channel_index].sampler
            time_accessor = glb.accessors[animation.samplers[curr_sampler_index].input]
            num_max_frames = max(num_max_frames , np.uint32(time_accessor.max[0]*time_mult) )
        print('NUM FRAMES: '+str(num_max_frames))
        
        # Init Animation channels
        all_animation_channels = []
        for i in range(len(node_list)):
            all_animation_channels.append(AnimationChannel(num_max_frames))
        
        # Read Animation channels
        for curr_channel_index in range(num_animation_channels):
            
            curr_animation_type = animation.channels[curr_channel_index].target.path
            curr_node = animation.channels[curr_channel_index].target.node   
                    
            curr_sampler_index = animation.channels[curr_channel_index].sampler
            time_accessor = glb.accessors[animation.samplers[curr_sampler_index].input]
            value_accessor = glb.accessors[animation.samplers[curr_sampler_index].output]
            
            n_time = time_accessor.count
            
            bv_time = glb.bufferViews[time_accessor.bufferView]
            bv_value = glb.bufferViews[value_accessor.bufferView]
            
            data_time = glb._glb_data[bv_time.byteOffset : bv_time.byteOffset + bv_time.byteLength]
            data_value = glb._glb_data[bv_value.byteOffset : bv_value.byteOffset + bv_value.byteLength]

            time_array = (time_mult * np.frombuffer(data_time, dtype=np.float32)).astype(np.uint32)
            
            value_array = np.frombuffer(data_value, dtype=np.float32)
            if value_accessor.type == 'VEC3':
                value_array = np.reshape(value_array, (-1, 3))
            elif value_accessor.type == 'VEC4':
                value_array = np.reshape(value_array, (-1, 4))
            
            if curr_animation_type == 'translation':
                for t in all_animation_channels[curr_node].complete_frames_time:
                    if isValueIsInArray(time_array,t):
                        # Get Value
                        currIndex = time_array==t
                        # Transform Rotation
                        RotImport = Rot.from_quat([0,0,0,1])          
                        if hasattr(glb.nodes[curr_node], 'rotation') and glb.nodes[curr_node].rotation != None:
                            RotImport = Rot.from_quat(glb.nodes[curr_node].rotation).inv()
                        RotMatrixImport = RotImport.as_matrix()
                        # Transform Translation
                        TransImport = [0,0,0]         
                        if hasattr(glb.nodes[curr_node], 'translation') and glb.nodes[curr_node].translation != None:
                            TransImport = glb.nodes[curr_node].translation
                        TransCurr = value_array[currIndex][0]
                        TransResult = TransCurr
                        if hasattr(glb.nodes[curr_node], 'translation') and glb.nodes[curr_node].translation != None and len(TransImport)>0:
                            TransResult = TransCurr - TransImport
                        # Save Translation
                        all_animation_channels[curr_node].complete_frames_trans[:,t-1] = np.matmul(RotMatrixImport,TransResult)
                    else:
                        # Get Interpolation
                        prevIndex = time_array<t
                        nextIndex = time_array>t
                        prevTime = time_array[prevIndex][0]
                        nextTime = time_array[nextIndex][0]
                        prevValue = value_array[prevIndex][0]
                        nextValue = value_array[nextIndex][0]
                        interpolationValue = getInterpolationValue(t,prevTime,nextTime)
                        interpolatedPoint = lerpPoint(prevValue, nextValue, interpolationValue)
                        # Transform Rotation
                        RotImport = Rot.from_quat([0,0,0,1])          
                        if hasattr(glb.nodes[curr_node], 'rotation') and glb.nodes[curr_node].rotation != None:
                            RotImport = Rot.from_quat(glb.nodes[curr_node].rotation).inv()
                        RotMatrixImport = RotImport.as_matrix()
                        # Transform Translation
                        TransImport = [0,0,0]         
                        if hasattr(glb.nodes[curr_node], 'translation') and glb.nodes[curr_node].translation != None:
                            TransImport = glb.nodes[curr_node].translation
                        TransCurr = interpolatedPoint.T
                        TransResult = TransCurr
                        if hasattr(glb.nodes[curr_node], 'translation') and glb.nodes[curr_node].translation != None and len(TransImport)>0:
                            TransResult = TransCurr - TransImport
                        # Save Translation
                        all_animation_channels[curr_node].complete_frames_trans[:,t-1] = np.matmul(RotMatrixImport,TransResult)
                
            elif curr_animation_type == 'scale':
                for t in all_animation_channels[curr_node].complete_frames_time:
                    if isValueIsInArray(time_array,t):
                        # Get Value
                        currIndex = time_array==t
                        # SWAP Y and Z
                        currScaleValue = value_array[currIndex][0]
                        # Save Scale
                        all_animation_channels[curr_node].complete_frames_scale[:,t-1] = currScaleValue
                    else:
                        # Get Interpolation
                        prevIndex = time_array<t
                        nextIndex = time_array>t
                        prevTime = time_array[prevIndex][0]
                        nextTime = time_array[nextIndex][0]
                        prevValue = value_array[prevIndex][0]
                        nextValue = value_array[nextIndex][0]
                        interpolationValue = getInterpolationValue(t,prevTime,nextTime)
                        interpolatedPoint = lerpPoint(prevValue, nextValue, interpolationValue)
                        # SWAP Y and Z
                        currScaleValue = interpolatedPoint.T
                        # Save Scale
                        all_animation_channels[curr_node].complete_frames_scale[:,t-1] = currScaleValue
                        
            elif curr_animation_type == 'rotation':
                for t in all_animation_channels[curr_node].complete_frames_time:
                    if isValueIsInArray(time_array,t):
                        # Get Value
                        currIndex = time_array==t
                        # Transform Rotation
                        RotImport = Rot.from_quat([0,0,0,1])          
                        if hasattr(glb.nodes[curr_node], 'rotation') and glb.nodes[curr_node].rotation != None:
                            RotImport = Rot.from_quat(glb.nodes[curr_node].rotation).inv()
                        RotCurr = Rot.from_quat(value_array[currIndex][0])
                        RotResult = +1*((RotImport * RotCurr).as_quat())
                        # SWAP Y and Z
                        currRotValue = RotResult
                        # Save Rotation
                        all_animation_channels[curr_node].complete_frames_rot[:,t-1] = currRotValue
                    else:
                        # Get Interpolation
                        prevIndex = time_array<t
                        nextIndex = time_array>t
                        prevTime = time_array[prevIndex][0]
                        nextTime = time_array[nextIndex][0]
                        prevValue = value_array[prevIndex][0]
                        nextValue = value_array[nextIndex][0]
                        interpolationValue = getInterpolationValue(t,prevTime,nextTime)
                        interpolatedPoint = slerpQuat(prevValue, nextValue, interpolationValue)
                        # Transform Rotation
                        RotImport = Rot.from_quat([0,0,0,1])          
                        if hasattr(glb.nodes[curr_node], 'rotation') and glb.nodes[curr_node].rotation != None:
                            RotImport = Rot.from_quat(glb.nodes[curr_node].rotation).inv()
                        RotCurr = Rot.from_quat(interpolatedPoint.T)
                        RotResult = +1*((RotImport * RotCurr).as_quat())
                        # SWAP Y and Z
                        currRotValue = RotResult
                        # Save Rotation
                        all_animation_channels[curr_node].complete_frames_rot[:,t-1] = currRotValue
            
        # Save ANIM file
        file_out = open(name_anim, "wb")
        # 1) NUM JOINTS
        file_out.write(ctypes.c_uint32(len(node_list)))
        # 2) Iterate over all animation joints
        for curr_node in node_list:

            # number of frames
            file_out.write(ctypes.c_uint32(all_animation_channels[curr_node].numFrames))
            for i in range(all_animation_channels[curr_node].numFrames):
                # time
                file_out.write(ctypes.c_float(all_animation_channels[curr_node].complete_frames_time[i]))
                # translation
                file_out.write(ctypes.c_float(all_animation_channels[curr_node].complete_frames_trans[0,i]))
                file_out.write(ctypes.c_float(all_animation_channels[curr_node].complete_frames_trans[1,i]))
                file_out.write(ctypes.c_float(all_animation_channels[curr_node].complete_frames_trans[2,i]))
                # scale
                file_out.write(ctypes.c_float(all_animation_channels[curr_node].complete_frames_scale[0,i]))
                file_out.write(ctypes.c_float(all_animation_channels[curr_node].complete_frames_scale[1,i]))
                file_out.write(ctypes.c_float(all_animation_channels[curr_node].complete_frames_scale[2,i]))
                # rotation
                file_out.write(ctypes.c_float(all_animation_channels[curr_node].complete_frames_rot[0,i]))
                file_out.write(ctypes.c_float(all_animation_channels[curr_node].complete_frames_rot[1,i]))
                file_out.write(ctypes.c_float(all_animation_channels[curr_node].complete_frames_rot[2,i]))
                file_out.write(ctypes.c_float(all_animation_channels[curr_node].complete_frames_rot[3,i]))         
        file_out.close()
    

print('READING GLB ...')

# READ FILE
glb = pygltflib.GLTF2().load(glb_filename)

# Read BINMESH
if binmesh_file_set:
    read_binmesh(glb,0,binmesh_filename)

if json_file_set:
    out_json_file = open(json_filename, "w")
    out_json_file.write(glb.gltf_to_json())

# Read animation
read_skel_anim(glb,0,fps_blender,skel_file_set,skel_filename,anim_file_set,anim_filename,txt_file_set,txt_filename)