import numpy as np
import tensorflow as tf
from vedo import *
import vtk
import time

def get_cluster_fast(rads, thresh = 0.5, mnp = 100, max_buffer = 0.5):
    """NEW VERSION using TF operations
    max_buffer = 0.5 #0.2
    """
    before = time.time()

    #fix dimensions
    if len(tf.shape(rads)) < 2:
        rads = rads[:,None]

    OG_rads = rads #hold on to OG rads
    mask = tf.cast(tf.math.equal(rads, 0), tf.float32)*1000
    rads = rads + mask

    #sort in ascending order for each column in tensor
    top_k = tf.math.top_k(tf.transpose(rads), k = tf.shape(rads)[0])
    rads = tf.transpose(tf.gather(tf.transpose(rads), top_k[1], batch_dims = 1))
    rads = tf.reverse(rads, axis = tf.constant([0]))

    # calculate the forward difference between neighboring points
    z = tf.zeros([1, tf.shape(rads)[1].numpy()])
    shifted = tf.concat((rads[1:], z), axis = 0)
    diff = shifted - rads

    # #find where difference jumps
    jumps = tf.where(diff > thresh)

    #----------------------------------------------------------------------
    #TO FIX BUG THAT PREVENTS VOXELS FROM BEING FORMED AROUND VERY TIGHT DISTINCT CLUSETERS OF POINTS (8/9/22)
    #       This is happening becuase we are adding in 0 as a first element only to spikes(?) that already have jumps!
    #get indexes of all used spikes
    used = jumps[:,1][None,:]
    biggest = tf.math.reduce_max(used, axis = 1).numpy()[0]
    all_spikes = tf.cast(tf.linspace(0,biggest,biggest+1), tf.int64)[None,:] #list all spikes total

    #find differnce
    missing = tf.sets.difference(all_spikes, used).values[None,:]
    zero = tf.constant(0, dtype = tf.float32)
    ends = tf.math.reduce_sum(tf.cast(tf.not_equal(OG_rads, zero), tf.int64), axis = 0) #correct

    test = tf.gather(ends, missing[0])  #get index of last element of missing jump section
    # print("\n test", test)
    z = test[None,:]
    z -= 2 #fixes indexing bug
    missing = tf.transpose(tf.concat((z, missing), axis = 0))

    #concat missing stuff back at the end of jumps
    jumps = tf.concat((jumps, missing), axis = 0)
    #----------------------------------------------------------------------

    jumps_temp = tf.gather(jumps, tf.argsort(jumps[:,1]), axis=0) #reorder based on index
    y, idx = tf.unique(jumps_temp[:,1]) #test

    jumps_rag = tf.RaggedTensor.from_value_rowids(jumps_temp[:,0], jumps_temp[:,1])

    # append 0 to beginning of each ragged elemet of jumps_rag
    zeros = tf.zeros(tf.shape(jumps_rag)[0])[:,None]
    zeros = tf.cast(tf.RaggedTensor.from_tensor(zeros), tf.int64)
    jumps_rag = tf.concat([zeros.with_row_splits_dtype(tf.int64), jumps_rag.with_row_splits_dtype(tf.int64)], axis = 1)

    #get num points between each jump 
    npts_between_jumps = tf.experimental.numpy.diff(jumps_rag.to_tensor())

    #flag spikes where all npts_between_jumps are less than mnp
    biggest_jump = tf.math.reduce_max(npts_between_jumps, axis = 1)
    # mnp = 100 #minumum number of points per cluster (defined in ICET class)
    good_clusters = tf.cast(tf.math.greater(biggest_jump, mnp), tf.int32)

    #get idx within jumps_rag corresponding to first sufficiently large jump
    big_enough = tf.cast(tf.math.greater(npts_between_jumps, mnp), tf.int32)
    first_big_enough = tf.math.argmax(big_enough, axis = 1)

    #simple way-- just use radial measurements of inner and outermost points in cluster
    # #--------------------------------------------------------------------------------------------------------
    # inner_idx = tf.gather(jumps_rag.to_tensor(), first_big_enough, batch_dims=1) + 1
    # inner  = tf.gather(tf.transpose(rads), inner_idx, batch_dims=1)
    # outer_idx = tf.gather(jumps_rag.to_tensor(), first_big_enough + 1, batch_dims=1)
    # outer  = tf.gather(tf.transpose(rads), outer_idx, batch_dims=1)
    # #--------------------------------------------------------------------------------------------------------

    #as described in paper
    #--------------------------------------------------------------------------------------------------------
    inner_idx = tf.gather(jumps_rag.to_tensor(), first_big_enough, batch_dims=1) + 1
    inner_radii  = tf.gather(tf.transpose(rads), inner_idx, batch_dims=1)
    #get radial distance of closest point on near side of cluster
    next_inner_idx = tf.gather(jumps_rag.to_tensor(), tf.nn.relu(first_big_enough - 1), batch_dims=1) -1 #think the -1 fixes bug (12/11)
    next_inner_radii = tf.gather(tf.transpose(rads), tf.nn.relu(next_inner_idx), batch_dims=1) 

    #will be zero when inner idx occurs on first element of spike, otherwise correct soln
    inner_skip_dist = inner_radii - next_inner_radii
    #of these nonzero distances, some are smaller than max_buffer -> leave as is, all else set to max_buffer
    too_big = tf.cast(tf.math.less(inner_skip_dist*2, max_buffer), tf.float32)
    # inner_skip_dist = inner_skip_dist*too_big + (1-too_big)*max_buffer #was this
    inner_skip_dist = inner_skip_dist*too_big + too_big*max_buffer #fixes things!
    temp = tf.cast(tf.math.equal(inner_skip_dist, 0), tf.float32)*max_buffer #set all others to max_buffer
    inner = inner_radii - inner_skip_dist - temp

    #repeat similar process for outer limits of each cell
    outer_idx = tf.gather(jumps_rag.to_tensor(), tf.nn.relu(first_big_enough + 1), batch_dims=1) - 1
    outer_radii  = tf.gather(tf.transpose(rads), tf.nn.relu(outer_idx), batch_dims=1)
    next_outer_idx = tf.gather(jumps_rag.to_tensor(), first_big_enough + 1, batch_dims=1) +1
    next_outer_radii = tf.gather(tf.transpose(rads), next_outer_idx, batch_dims=1) 

    outer_skip_dist = next_outer_radii - outer_radii
    too_big = tf.cast(tf.math.less(outer_skip_dist*2, max_buffer), tf.float32)
    outer_skip_dist = outer_skip_dist*too_big + (1-too_big)*max_buffer
    outer = outer_radii + outer_skip_dist
    #--------------------------------------------------------------------------------------------------------

    # bounds = tf.convert_to_tensor(np.array([inner, outer]).T)
    bounds = tf.concat((inner[:,None], outer[:,None]), axis = 1)
    bounds = tf.cast(good_clusters[:,None], tf.float32) * bounds #suppress cells with no good clusters

    # print("\n getting bounds took", time.time() - before,"seconds")

    return bounds


def get_cluster(rads, thresh = 0.5, mnp = 100): #mnp = 50, thresh = 0.2
    """ Identifies radial bounds which contain the first cluster in a spike 
            that is closest to the ego-vehicle 
        
        rads = tensor containing radii of points in each spike
        thresh = must be this close to nearest neighbor to be considered part of a cluster
        mnp = minimum number of points a cluster must contain to be considered
            """
    before = time.time()

    #TODO: try dymacally lowering <max_buffer> value as algorithm progresses
    max_buffer = 0.2 #was this
    # max_buffer = 0.5 #test

    #fix dimensions
    if len(tf.shape(rads)) < 2:
        rads = rads[:,None]

    OG_rads = rads #hold on to OG rads
    #replace all zeros in rads (result of converting ragged -> standard tensor) with some arbitrarily large value
    mask = tf.cast(tf.math.equal(rads, 0), tf.float32)*1000
    rads = rads + mask
    # print(rads)

    #sort in ascending order for each column in tensor
    top_k = tf.math.top_k(tf.transpose(rads), k = tf.shape(rads)[0])
    # print("\n top_k \n", top_k[1])
    rads = tf.transpose(tf.gather(tf.transpose(rads), top_k[1], batch_dims = 1))
    rads = tf.reverse(rads, axis = tf.constant([0]))
    # print("rads \n", rads)


    # calculate the forward difference between neighboring points
    z = tf.zeros([1, tf.shape(rads)[1].numpy()])
    shifted = tf.concat((rads[1:], z), axis = 0)
    diff = shifted - rads
    # diff = tf.math.abs(rads - shifted) #debug 6/9/22
    # print("\n diff \n", diff)

    # #find where difference jumps
    jumps = tf.where(diff > thresh)
    # print("\n jumps \n", jumps) #[idx of jump, which spike is jumping]

    #----------------------------------------------------------------------
    #TO FIX BUG THAT PREVENTS VOXELS FROM BEING FORMED AROUND VERY TIGHT DISTINCT CLUSETERS OF POINTS (8/9/22)
    #       This is happening becuase we are adding in 0 as a first element only to spikes(?) that already have jumps!
   
    #get indexes of all used spikes
    used = jumps[:,1][None,:]
    # print("used", used)
    biggest = tf.math.reduce_max(used, axis = 1).numpy()[0]
    # print("biggest", biggest)
    all_spikes = tf.cast(tf.linspace(0,biggest,biggest+1), tf.int64)[None,:] #list all spikes total
    # print("all_spikes", all_spikes)

    #find differnce
    missing = tf.sets.difference(all_spikes, used).values[None,:]
    # print("\n missing", missing)
    # z = tf.zeros(tf.shape(missing), dtype = tf.int64) #wrong...
    # z = 51*tf.ones(tf.shape(missing), dtype = tf.int64) #wrong...
    # print("z", z)

    #z should be this...
    # print("\n OG_rads", OG_rads)
    # ends = tf.math.argmax(OG_rads, axis = 0) #wrong -> not max arg, last nonzero argument!!
    zero = tf.constant(0, dtype = tf.float32)
    ends = tf.math.reduce_sum(tf.cast(tf.not_equal(OG_rads, zero), tf.int64), axis = 0) #correct
    # print("\n ends", ends)

    test = tf.gather(ends, missing[0])  #get index of last element of missing jump section
    # print("\n test", test)
    z = test[None,:]
    z -= 2 #fixes indexing bug
    # print("z", z)

    missing = tf.transpose(tf.concat((z, missing), axis = 0))
    # print(missing)

    #concat missing stuff back at the end of jumps
    jumps = tf.concat((jumps, missing), axis = 0)
    # print("\n jumps after fix", jumps)
    #----------------------------------------------------------------------


    #find where the first large cluster occurs in each spike
    #   using numpy here because we're not working with the full dataset and 
    #   it's easier if we use in place operations (but way slower!!!)
    bounds = np.zeros([tf.shape(rads)[1].numpy(), 2])
    for i in range(tf.shape(rads)[1].numpy()):

        #get the indices of jumps for the ith spike
        jumps_i = tf.gather(jumps, tf.where(jumps[:,1] == i))[:,0].numpy()
        # jumps_i = np.append(np.zeros([1,2], dtype = np.int32), jumps_i, axis = 0)#need to add zeros to the beginning
        jumps_i = np.append(tf.constant([[0,i]], dtype = np.int32), jumps_i, axis = 0) #test 8/10/22

        
        # print("jumps_i", i, " \n", jumps_i)  

        last = 0
        count = 1
        while True:

            #degbug
            # print(tf.shape(jumps_i))
            if tf.shape(jumps_i)[0] < 2:
                bounds[i,:] = 0
                break

            #check and see if this jump contains a sufficient number of points
            if jumps_i[count ,0] - last > mnp:
                # print(last, count)

                #set bounds at edges of cluster
                bounds[i, 0] = rads[jumps_i[count - 1, 0] + 1, i]
                bounds[i, 1] = rads[jumps_i[count, 0], i] 

                # #extend cluster bounds halfway to next point
                # buffer_dist1 = (rads[jumps_i[count - 1, 0] , i] - rads[jumps_i[count - 1, 0] - 1, i]) / 2
                # # print("b1", buffer_dist1)
                # if abs(buffer_dist1) > max_buffer:
                #     buffer_dist1 = max_buffer
                # bounds[i, 0] =  rads[jumps_i[count - 1, 0] + 1, i] - buffer_dist1

                # buffer_dist2 = (rads[jumps_i[count, 0] + 1, i] - rads[jumps_i[count, 0], i]) / 2
                # # print("b2", buffer_dist2)
                # if buffer_dist2 > max_buffer:
                #     buffer_dist2 = max_buffer
                # bounds[i, 1] =  rads[jumps_i[count, 0], i] + buffer_dist2
                
                break 

            last = jumps_i[count, 0]
            count += 1

            #if no useful clusters appear
            if count == tf.shape(jumps_i)[0]:
                bounds[i, :] = 0
                break

    bounds = tf.convert_to_tensor(bounds)
    # print("\n bounds", bounds)

    print("\n getting cluster took", time.time() - before,"seconds !!!")
    return(bounds)


def R2Euler(mat):
    """determines euler angles from euler rotation matrix"""

    if len( tf.shape(mat) ) == 2:
        mat = mat[None, :, :]

    R_sum = np.sqrt(( mat[:,0,0]**2 + mat[:,0,1]**2 + mat[:,1,2]**2 + mat[:,2,2]**2 ) / 2)

    phi = np.arctan2(-mat[:,1,2],mat[:,2,2])
    theta = np.arctan2(mat[:,0,2], R_sum)
    psi = np.arctan2(-mat[:,0,1], mat[:,0,0])

    angs = np.array([phi, theta, psi])
    return angs

def R_tf(angs):
    """generates rotation matrix using euler angles
    angs = tf.constant(phi, theta, psi) (aka rot about (x,y,z))
            can be single set of angles or batch for multiple cells
    """

    if len(tf.shape(angs)) == 1:
        angs = angs[None,:]

    phi = angs[:,0]
    theta = angs[:,1]
    psi = angs[:,2]

    mat = tf.Variable([[cos(theta)*cos(psi), sin(psi)*cos(phi) + sin(phi)*sin(theta)*cos(psi), sin(phi)*sin(psi) - sin(theta)*cos(phi)*cos(psi)],
                       [-sin(psi)*cos(theta), cos(phi)*cos(psi) - sin(phi)*sin(theta)*sin(psi), sin(phi)*cos(psi) + sin(theta)*sin(psi)*cos(phi)],
                       [sin(theta), -sin(phi)*cos(theta), cos(phi)*cos(theta)]
                        ])

    mat = tf.transpose(mat, [2, 0, 1])
    mat = tf.squeeze(mat)
    return mat

def jacobian_tf(p_point, angs):
    """calculates jacobian for point using TensorFlow
        angs = tf.constant[phi, theta, psi] aka (x,y,z)"""

    phi = angs[0]
    theta = angs[1]
    psi = angs[2]

    #correct method using tf.tile
    eyes = tf.tile(-tf.eye(3), [tf.shape(p_point)[1] , 1])

    # (deriv of R() wrt phi).dot(p_point)
    #   NOTE: any time sin/cos operator is used, output will be 1x1 instead of constant (not good)
    Jx = tf.tensordot(tf.Variable([[tf.constant(0.), (-sin(psi)*sin(phi) + cos(phi)*sin(theta)*cos(psi)), (cos(phi)*sin(psi) + sin(theta)*sin(phi)*cos(psi))],
                                   [tf.constant(0.), (-sin(phi)*cos(psi) - cos(phi)*sin(theta)*sin(psi)), (cos(phi)*cos(psi) - sin(theta)*sin(psi)*sin(phi))], 
                                   [tf.constant(0.), (-cos(phi)*cos(theta)), (-sin(phi)*cos(theta))] ]), p_point, axes = 1)

    # (deriv of R() wrt theta).dot(p_point)
    Jy = tf.tensordot(tf.Variable([[(-sin(theta)*cos(psi)), (cos(theta)*sin(phi)*cos(psi)), (-cos(theta)*cos(phi)*cos(psi))],
                                   [(sin(psi)*sin(theta)), (-cos(theta)*sin(phi)*sin(psi)), (cos(theta)*sin(psi)*cos(phi))],
                                   [(cos(theta)), (sin(phi)*sin(theta)), (-sin(theta)*cos(phi))] ]), p_point, axes = 1)

    Jz = tf.tensordot(tf.Variable([[(-cos(theta)*sin(psi)), (cos(psi)*cos(phi) - sin(phi)*sin(theta)*sin(psi)), (cos(psi)*sin(phi) + sin(theta)*cos(phi)*sin(psi)) ],
                                       [(-cos(psi)*cos(theta)), (-sin(psi)*cos(phi) - sin(phi)*sin(theta)*cos(psi)), (-sin(phi)*sin(psi) + sin(theta)*cos(psi)*cos(phi))],
                                       [tf.constant(0.),tf.constant(0.),tf.constant(0.)]]), p_point, axes = 1)

    Jx_reshape = tf.reshape(tf.transpose(Jx), shape = (tf.shape(Jx)[0]*tf.shape(Jx)[1],1))
    Jy_reshape = tf.reshape(tf.transpose(Jy), shape = (tf.shape(Jy)[0]*tf.shape(Jy)[1],1))
    Jz_reshape = tf.reshape(tf.transpose(Jz), shape = (tf.shape(Jz)[0]*tf.shape(Jz)[1],1))

    J = tf.concat([eyes, Jx_reshape, Jy_reshape, Jz_reshape], axis = 1) #was this

    return J

class Ell(Mesh):
    """
    Build a 3D ellipsoid centered at position `pos`.

    |projectsphere|

    |pca| |pca.py|_
    """
    def __init__(self, pos=(0, 0, 0), axis1= 1, axis2 = 2, axis3 = 3, angs = np.array([0,0,0]),
                 c="cyan4", alpha=1, res=24):

        self.center = pos
        self.va_error = 0
        self.vb_error = 0
        self.vc_error = 0
        self.axis1 = axis1
        self.axis2 = axis2
        self.axis3 = axis3
        self.nr_of_points = 1 # used by pcaEllipsoid

        if utils.isSequence(res):
            res_t, res_phi = res
        else:
            res_t, res_phi = 2*res, res

        elliSource = vtk.vtkSphereSource()
        elliSource.SetThetaResolution(res_t)
        elliSource.SetPhiResolution(res_phi)
        elliSource.Update()
        l1 = axis1
        l2 = axis2
        l3 = axis3
        self.va = l1
        self.vb = l2
        self.vc = l3
        axis1 = 1
        axis2 = 1
        axis3 = 1
        angle = angs[0] #np.arcsin(np.dot(axis1, axis2))
        theta = angs[1] #np.arccos(axis3[2])
        phi =  angs[2] #np.arctan2(axis3[1], axis3[0])

        t = vtk.vtkTransform()
        t.PostMultiply()
        t.Scale(l1, l2, l3)

        #needed theta and angle to be negative before messing with E_xz, E_yz...
        t.RotateZ(np.rad2deg(phi))
        t.RotateY(-np.rad2deg(theta)) #flipped sign here 5/19
        t.RotateX(-np.rad2deg(angle)) #flipped sign here 5/19
        
        tf = vtk.vtkTransformPolyDataFilter()
        tf.SetInputData(elliSource.GetOutput())
        tf.SetTransform(t)
        tf.Update()
        pd = tf.GetOutput()
        self.transformation = t

        Mesh.__init__(self, pd, c, alpha)
        self.phong()
        self.GetProperty().BackfaceCullingOn()
        self.SetPosition(pos)
        self.Length = -np.array(axis1) / 2 + pos
        self.top = np.array(axis1) / 2 + pos
        self.name = "Ell"