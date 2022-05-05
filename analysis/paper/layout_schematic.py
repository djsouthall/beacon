import json
import sys
import os
import copy
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
plt.ion()
from beacon.tools.angle_annotation import AngleAnnotation
import beacon.tools.info as info
from beacon.tools.config_reader import *
import pymap3d as pm
import numpy
from scipy.linalg import lstsq
import datetime


import matplotlib.patheffects as PathEffects
import inspect


def fitPlaneGetNorm(xyz,verbose=False):
    '''
    Given an array of similarly lengthed x,y,z data, this will fit a plane to the data, and then
    determine the normal vector to that plane and return it.  The xyz data should be stacked such that
    the first column is x, second is y, and third is z.
    '''
    try:
        x = xyz[:,0]
        y = xyz[:,1]
        z = xyz[:,2]

        A = numpy.matrix(numpy.vstack((x,y,numpy.ones_like(x))).T)
        b = numpy.matrix(z).T
        fit, residual, rnk, s = lstsq(A, b)
        if verbose:
            print("solution: %f x + %f y + %f = z" % (fit[0], fit[1], fit[2]))
        plane_func = lambda _x, _y : fit[0]*_x + fit[1]*_y + fit[2]
        zero = numpy.array([0,0,plane_func(0,0)[0]]) #2 points in plane per vector, common point at 0,0 between the 2 vectors. 

        v0 = numpy.array([1,0,plane_func(1,0)[0]]) - zero
        v0 = v0/numpy.linalg.norm(v0)
        v1 = numpy.array([0,1,plane_func(0,1)[0]]) - zero
        v1 = v1/numpy.linalg.norm(v1)
        norm = numpy.cross(v0,v1)
        norm = norm/numpy.linalg.norm(norm)
        return norm
    except Exception as e:
        print('\nError in %s'%inspect.stack()[0][3])
        print(e)
        exc_type, exc_obj, exc_tb = sys.exc_info()
        fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
        print(exc_type, fname, exc_tb.tb_lineno)

def planeFromNorm(norm, p):
    norm = norm/numpy.linalg.norm(norm)
    z = lambda x,y : (- norm[0]*(x - p[0]) - norm[1]*(y - p[1]))/norm[2] + p[2]
    return z

def planeFromXYZ(xyz):
    plane_norm = fitPlaneGetNorm(xyz)
    return planeFromNorm(plane_norm, xyz[0,:])




if __name__ == '__main__':
    deploy_index = info.returnDefaultDeploy()

    deploy_index

    en_figsize=(16,16)
    eu_figsize=(16,9)
    combined_figsize = (16,16)


    mast_height=12*0.3048
    antenna_scale_factor=5
    mast_colors=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']


    origin, antennas_physical, antennas_phase_hpol, antennas_phase_vpol, cable_delays = configReader(deploy_index)
    antenna_scale_factor = int(antenna_scale_factor)

    box_height  = antenna_scale_factor * 0.11506
    box_width   = antenna_scale_factor * 0.06502
    box_color   = 'silver'
    mountain_color = 'silver'

    element_height = antenna_scale_factor * 1.40462 #This is the point to point extent of antenna
    element_width = 4*antenna_scale_factor * 0.009525

    # mode='phase_hpol'
    modes = ['phase_hpol']#['physical', 'phase_hpol', 'phase_vpol']
    for mode in modes:

        if mode == 'physical':
            ants = antennas_physical
        elif mode == 'phase_hpol':
            ants = antennas_phase_hpol
        elif mode == 'phase_vpol':
            ants = antennas_phase_vpol
        elif mode == 'all':
            print('Selected mode \'all\' not yet handled, using physical.')
            mode = 'physical'
            ants = antennas_physical
        
        ants = numpy.vstack((ants[0],ants[1],ants[2],ants[3]))
        # import pdb; pdb.set_trace()


        plane_equation = planeFromXYZ(ants)

        e_span = numpy.max(ants[:,0]) - numpy.min(ants[:,0])
        e_vals = numpy.linspace(min(ants[:,0]) - 0.1*e_span, max(ants[:,0]) + 0.1*e_span, 10)
        n_span = numpy.max(ants[:,1]) - numpy.min(ants[:,1])
        n_vals = numpy.linspace(min(ants[:,1]) - 0.1*n_span, max(ants[:,1]) + 0.1*n_span, 10)
        u_span = numpy.max(ants[:,2]) - numpy.min(ants[:,2])
        u_vals = numpy.linspace(min(ants[:,2]) - 0.1*u_span, max(ants[:,2]) + 0.1*u_span, 10)


        #import pdb; pdb.set_trace()
        
        figs = []
        axs = []
        names = []
        annotate_bar_fontsize = 12
        magnify_label_fontsize = 12
        angle_fontsize = 12
        label_fontsize = 16
        vector_length = 0.75*element_height

        if False:
            #Top down view
            names.append('EN')
            fig = plt.figure(figsize=en_figsize)
            figs.append(fig)
            ax1 = plt.gca()
            plt.axis('off')
            axs.append(ax1)
            ax1.axis('equal')
            
            #Add baselines
            for pair in [[0,1],[0,2],[0,3],[1,2],[1,3],[2,3]]:
                p1_xy = numpy.array([ants[pair[0]][0], ants[pair[0]][1]])
                p2_xy = numpy.array([ants[pair[1]][0], ants[pair[1]][1]])
                bbox=dict(fc="white", ec="none")
                ax1.annotate("", xy=p1_xy, xytext=p2_xy, textcoords=ax1.transData, arrowprops={"arrowstyle" : "-", "linestyle" : "--", "linewidth" : 0.5}, zorder = 0)
                #bbox=dict(fc="white", ec="none")
                text_coords = (3*p2_xy + 2*p1_xy) / 5 + numpy.array([0.5,0.5])
                ax1.text(text_coords[0], text_coords[1], '%i-%i'%(pair[0],pair[1]), bbox=bbox, ha="center", va="center", fontsize=10)
            
            for mast in range(4):
                ax1.add_patch(Rectangle((ants[mast][0] + box_width/2 - element_width/2, ants[mast][1] - element_height/2), width=element_width, height=element_height, facecolor=mast_colors[mast], edgecolor=mast_colors[mast], zorder = 10))
                ax1.add_patch(Rectangle((ants[mast][0],ants[mast][1] - box_height/2), width=box_width, height=box_height, facecolor=box_color, edgecolor='k', zorder = 50))
                plt.text(ants[mast][0] + 1, ants[mast][1], str(mast), fontsize=label_fontsize, zorder = 100)

            #Add horizontal label
            p1_xy = numpy.array([ants[1][0], ants[0][1] + element_height])
            p2_xy = numpy.array([ants[0][0], ants[0][1] + element_height])
            orientation = 'horizontal'
            if orientation == 'horizontal':
                dist = abs(p1_xy[0] - p2_xy[0])
                text_coords = ( (p2_xy[0] + p1_xy[0])/2 , p1_xy[1] )
            else:
                dist = abs(p1_xy[1] - p2_xy[1])
                text_coords = ( p1_xy[0] , (p2_xy[1] + p1_xy[1])/2 )

            ax1.annotate("", xy=p1_xy, xytext=p2_xy, textcoords=ax1.transData, arrowprops=dict(arrowstyle='<->'))
            ax1.annotate("", xy=p1_xy, xytext=p2_xy, textcoords=ax1.transData, arrowprops=dict(arrowstyle='|-|'))
            bbox=dict(fc="white", ec="none")
            ax1.text(text_coords[0], text_coords[1], "L=%0.1f m"%(dist), ha="center", va="center", bbox=bbox, fontsize=annotate_bar_fontsize, rotation = orientation)


            #Add vertical label
            p1_xy = numpy.array([ants[0][0] + element_height, ants[0][1]])
            p2_xy = numpy.array([ants[0][0] + element_height, ants[2][1]])
            orientation = 'vertical'
            if orientation == 'horizontal':
                dist = abs(p1_xy[0] - p2_xy[0])
                text_coords = ( (p2_xy[0] + p1_xy[0])/2 , p1_xy[1] )
            else:
                dist = abs(p1_xy[1] - p2_xy[1])
                text_coords = ( p1_xy[0] , (p2_xy[1] + p1_xy[1])/2 )

            ax1.annotate("", xy=p1_xy, xytext=p2_xy, textcoords=ax1.transData, arrowprops=dict(arrowstyle='<->'))
            ax1.annotate("", xy=p1_xy, xytext=p2_xy, textcoords=ax1.transData, arrowprops=dict(arrowstyle='|-|'))
            bbox=dict(fc="white", ec="none")
            ax1.text(text_coords[0], text_coords[1], "L=%0.1f m"%(dist), ha="center", va="center", bbox=bbox, fontsize=annotate_bar_fontsize, rotation = orientation)


            #Add axis label
            vector_origin = numpy.array([ants[3][0] - 1.5*element_height , ants[3][1]])
            plt.arrow(vector_origin[0],vector_origin[1], 0, vector_length,color='k',label='N',head_width = box_height)
            plt.text(vector_origin[0] + 0.75,vector_origin[1] + vector_length, 'N',fontsize = label_fontsize)
            plt.arrow(vector_origin[0],vector_origin[1], vector_length,0,color='k',label='E',head_width = box_height)
            plt.text(vector_origin[0] + vector_length,vector_origin[1] + 0.75, 'E',fontsize = label_fontsize)




            plt.tight_layout()
            plt.ylim(min(n_vals),max(n_vals) + element_height)
            plt.xlim(min(e_vals),max(e_vals))
            plt.text(0.01, 0.01, 'Antenna\'s Magnified %ix'%antenna_scale_factor, fontsize=magnify_label_fontsize, transform=ax1.transAxes)


        if False:
            names.append('EU')
            #Side view
            fig = plt.figure(figsize=eu_figsize)
            figs.append(fig)
            ax2 = plt.gca()
            #ax2.sharex(ax1)
            plt.axis('off')
            axs.append(ax2)
            ax2.axis('equal')





            #Plotting ground
            y1 = ants[1][2]#numpy.min((ants[1][2],ants[3][2])) #Uphill
            y2 = ants[0][2]#numpy.max((ants[0][2],ants[2][2])) #Downhill
            
            x1 = ants[1][0]#ants[[1,3][numpy.argmin((ants[1][2],ants[3][2]))]][0] #Uphill
            x2 = ants[0][0]#ants[[0,2][numpy.argmax((ants[0][2],ants[2][2]))]][0] #Downhill

            slope = ( y2 - y1 ) / ( x2 - x1 ) #Assumes current form of array
            intercept = ants[0][0] - mast_height#ants[[0,2][numpy.argmax((ants[0][2],ants[2][2]))]][2] - mast_height #The higher of the 2

            y = lambda x : slope * x + intercept 


            for mast in range(4):
                plt.plot((ants[mast][0],ants[mast][0]),(y(ants[mast][0]), ants[mast][2]), c='k',linewidth=2, zorder = 0)
                
                ax2.add_patch(Rectangle((ants[mast][0] + box_width/2 - element_width/2, ants[mast][2] - element_height/2), width=element_width, height=element_height, facecolor=mast_colors[mast], edgecolor=mast_colors[mast], zorder = 10))
                ax2.add_patch(Rectangle((ants[mast][0],ants[mast][2] - box_height/2), width=box_width, height=box_height, facecolor=box_color, edgecolor='k', zorder = 100))

                plt.text(ants[mast][0] + 1, ants[mast][2], str(mast), fontsize=label_fontsize)
                
            large_range = numpy.array([min(e_vals) - 100, max(e_vals) + 100])
            plt.fill_between(large_range, y(large_range), -100, facecolor=mountain_color)#, hatch='x')

            #Add vertical label
            p1_xy = numpy.array([ants[1][0] - element_height, ants[1][2]])
            p2_xy = numpy.array([ants[1][0] - element_height, ants[0][2]])
            orientation = 'vertical'
            if orientation == 'horizontal':
                dist = abs(p1_xy[0] - p2_xy[0])
                text_coords = ( (p2_xy[0] + p1_xy[0])/2 , p1_xy[1] )
            else:
                dist = abs(p1_xy[1] - p2_xy[1])
                text_coords = ( p1_xy[0] , (p2_xy[1] + p1_xy[1])/2 )

            ax2.annotate("", xy=p1_xy, xytext=p2_xy, textcoords=ax2.transData, arrowprops=dict(arrowstyle='<->'))
            ax2.annotate("", xy=p1_xy, xytext=p2_xy, textcoords=ax2.transData, arrowprops=dict(arrowstyle='|-|'))
            bbox=dict(fc='silver', ec="none")
            ax2.text(text_coords[0], text_coords[1], "L=%0.1f m"%(dist), ha="center", va="center", bbox=bbox, fontsize=annotate_bar_fontsize, rotation = orientation)

            length = 5
            angle_deg = numpy.rad2deg(abs(numpy.arctan(slope)))
            angle_rad = abs(numpy.arctan(slope))
            center = (numpy.mean(e_vals) - 0.5, y(numpy.mean(e_vals) - 0.5))
            p1 = (center[0] - length*numpy.cos(angle_rad),center[1] + length*numpy.sin(angle_rad))
            p2 = (center[0] - length, center[1])
            # import pdb; pdb.set_trace()

            plt.plot([center[0] , p1[0]], [center[1] , p1[1]],c='k')
            plt.plot([center[0] , p2[0]], [center[1] , p2[1]],c='k')

            ang = AngleAnnotation(center, p1, p2, ax=ax2, size=75, text= '%i'%(angle_deg) + r"$^\circ$" + '',text_kw=dict(fontsize=angle_fontsize),textposition='outside')
            
            #Add axis label
            vector_origin = numpy.array([ants[3][0] - 1.5*element_height , ants[0][2] - 10])
            plt.arrow(vector_origin[0],vector_origin[1], 0, vector_length,color='k',label='U',head_width = box_height)
            plt.text(vector_origin[0] + 0.75,vector_origin[1] + vector_length, 'U',fontsize = label_fontsize)
            plt.arrow(vector_origin[0],vector_origin[1], vector_length,0,color='k',label='E',head_width = box_height)
            plt.text(vector_origin[0] + vector_length,vector_origin[1] + 0.75, 'E',fontsize = label_fontsize)

            plt.tight_layout()
            plt.ylim(min(u_vals),max(u_vals))
            plt.xlim(min(e_vals) - element_height,max(e_vals))
            
            
            plt.text(0.01, 0.01, 'Antenna\'s Magnified %ix'%antenna_scale_factor, fontsize=magnify_label_fontsize, transform=ax2.transAxes)

        if True:

            #Top down view
            names.append('EN')
            fig = plt.figure(figsize=(6,9))
            figs.append(fig)
            ax1 = plt.gca()
            plt.axis('off')
            axs.append(ax1)
            ax1.axis('equal')
            
            #Add baselines
            for pair in [[0,1],[0,2],[0,3],[1,2],[1,3],[2,3]]:
                i = pair[0]
                j = pair[1]
                p1_xy = numpy.array([ants[i][0], ants[i][1]])
                p2_xy = numpy.array([ants[j][0], ants[j][1]])
                angle = numpy.rad2deg(numpy.arctan2(p2_xy[1]-p1_xy[1], p2_xy[0]-p1_xy[0]))

                slope = 180.0 - numpy.rad2deg(numpy.arctan2(ants[j][2]-ants[i][2], ants[j][0]-ants[i][0]))            


                midpoint = (p2_xy + p1_xy)/2.0
                total_distance = numpy.sqrt(numpy.sum((ants[i] - ants[j])**2))

                if 1 in pair and 2 in pair:
                    text_coords = midpoint + 5*numpy.array([ 1, numpy.tan(numpy.deg2rad(angle))])
                else:
                    text_coords = midpoint#(3*p2_xy + 2*p1_xy) / 5 + numpy.array([0.5,0.5])
                print(text_coords)

                print(pair)
                print(angle)
                if angle <= 180.0 and angle > 100:
                    angle -= 180.0
                elif angle < -100 and angle >= -270:
                    angle += 180
                print(pair)
                print(angle)



                #ax1.transData
                ax1.annotate("", xy=p1_xy, xytext=p2_xy, textcoords="data", arrowprops={"arrowstyle" : "-", "linestyle" : "--", "linewidth" : 0.5}, zorder = 0)



                #bbox=dict(fc="white", ec="none")
                print(midpoint)
                bbox=dict(fc="white",boxstyle="round", ec="k", alpha=1.0)
                ax1.text(text_coords[0], text_coords[1], '%0.1f m'%(total_distance), bbox=bbox, ha="center", va="center", fontsize=14, rotation=angle)
            
            for mast in range(4):
                ax1.add_patch(Rectangle(
                            (ants[mast][0] + box_width/2 - element_width/2, ants[mast][1] - element_height/2),
                            width=element_width, height=element_height, 
                            facecolor=mast_colors[mast], edgecolor=mast_colors[mast], zorder = 10))

                #Add VPol component
                rotation_angle_deg=75
                ax1.add_patch(Rectangle(
                            (ants[mast][0] + box_width/2 - element_width/2 + element_height*numpy.cos(numpy.deg2rad(rotation_angle_deg))/2, ants[mast][1] - box_height/2 - element_width/2),
                            width=element_width, height=element_height*numpy.cos(numpy.deg2rad(rotation_angle_deg)), 
                            angle=rotation_angle_deg, capstyle='round',joinstyle='round',
                            facecolor=mast_colors[mast], edgecolor=mast_colors[mast], zorder = 10))


                ax1.add_patch(Rectangle((ants[mast][0],ants[mast][1] - box_height/2),
                            width=box_width, height=box_height, facecolor=box_color, edgecolor='k', zorder = 50))

                dl = 1
                if mast == 0:
                    dx, dy = dl, dl
                    va = 'bottom'
                    ha = 'left'
                elif mast == 1:
                    dx, dy = -dl, dl
                    va = 'bottom'
                    ha = 'right'

                elif mast == 2:
                    dx, dy = dl, -dl
                    va = 'top'
                    ha = 'left'
                elif mast == 3:
                    dx, dy = -dl, -dl
                    va = 'top'
                    ha = 'right'

                # t = str(mast) + ', U = %0.1f m'%(ants[mast][2])
                t = str(mast)
                plt.text(ants[mast][0] + dx, ants[mast][1] + dy,
                    t,
                    c=mast_colors[mast],
                    weight='bold',
                    va=va, ha=ha,
                    fontsize=label_fontsize, zorder = 100)

            if False:
                #Add horizontal label
                p1_xy = numpy.array([ants[1][0], ants[0][1] + element_height])
                p2_xy = numpy.array([ants[0][0], ants[0][1] + element_height])
                orientation = 'horizontal'
                if orientation == 'horizontal':
                    dist = abs(p1_xy[0] - p2_xy[0])
                    text_coords = ( (p2_xy[0] + p1_xy[0])/2 , p1_xy[1] )
                else:
                    dist = abs(p1_xy[1] - p2_xy[1])
                    text_coords = ( p1_xy[0] , (p2_xy[1] + p1_xy[1])/2 )

                ax1.annotate("", xy=p1_xy, xytext=p2_xy, textcoords=ax1.transData, arrowprops=dict(arrowstyle='|-|'))

                ax1.annotate("", xy=p1_xy, xytext=p2_xy, textcoords=ax1.transData, arrowprops=dict(arrowstyle='<->'))
                ax1.annotate("", xy=p1_xy, xytext=p2_xy, textcoords=ax1.transData, arrowprops=dict(arrowstyle='|-|'))

                bbox=dict(fc="white",boxstyle="round", ec="k", alpha=1.0)
                ax1.text(text_coords[0], text_coords[1], "L=%0.1f m"%(dist), ha="center", va="center", bbox=bbox, fontsize=annotate_bar_fontsize, rotation = orientation)
            
                #Add vertical label
                p1_xy = numpy.array([ants[0][0] + element_height, ants[0][1]])
                p2_xy = numpy.array([ants[0][0] + element_height, ants[2][1]])
                orientation = 'vertical'
                if orientation == 'horizontal':
                    dist = abs(p1_xy[0] - p2_xy[0])
                    text_coords = ( (p2_xy[0] + p1_xy[0])/2 , p1_xy[1] )
                else:
                    dist = abs(p1_xy[1] - p2_xy[1])
                    text_coords = ( p1_xy[0] , (p2_xy[1] + p1_xy[1])/2 )

                ax1.annotate("", xy=p1_xy, xytext=p2_xy, textcoords=ax1.transData, arrowprops=dict(arrowstyle='<->'))
                ax1.annotate("", xy=p1_xy, xytext=p2_xy, textcoords=ax1.transData, arrowprops=dict(arrowstyle='|-|'))
                bbox=dict(fc="white",boxstyle="round", ec="k", alpha=1.0)
                ax1.text(text_coords[0]+1, text_coords[1]+1, "L=%0.1f m"%(dist), ha="center", va="center", bbox=bbox, fontsize=annotate_bar_fontsize, rotation = orientation)


            if True:
                #Add axis label
                # vector_origin = numpy.array([ants[3][0] - 1.5*element_height , ants[3][1]])
                #vector_origin = numpy.array([ants[1][0] + 1*element_height , ants[0][1]-5])
                vector_origin = (-32, -2)
                plt.arrow(vector_origin[0],vector_origin[1], 0, vector_length,color='k',label='N',head_width = box_height)
                plt.text(vector_origin[0] + 0.75,vector_origin[1] + vector_length, 'N',fontsize = label_fontsize)
                plt.arrow(vector_origin[0],vector_origin[1], vector_length,0,color='k',label='E',head_width = box_height)
                plt.text(vector_origin[0] + vector_length,vector_origin[1] + 0.75, 'E',fontsize = label_fontsize)

                plt.ylim(min(n_vals),max(n_vals) + element_height)
                plt.xlim(min(e_vals),max(e_vals))
                #plt.text(0.01, 0.01, 'Antenna\'s Magnified %ix'%antenna_scale_factor, fontsize=magnify_label_fontsize, transform=ax1.transAxes)

                plane_E, plane_N = numpy.meshgrid(numpy.linspace(min(e_vals),max(e_vals), 100), numpy.linspace(min(n_vals),max(n_vals) + element_height, 100))
                plane_U = plane_equation(plane_E, plane_N)

                # plt.contourf(plane_E, plane_N, plane_U, levels = 20, cmap='Greys',alpha=0.4)#numpy.sort(ants[:,2])

                plt.ylim(min(n_vals),max(n_vals) + element_height)
                plt.xlim(min(e_vals),max(e_vals))

            if True:
                #Add axis label
                # vector_origin = numpy.array([ants[3][0] - 1.5*element_height , ants[3][1]])
                #vector_origin = numpy.array([ants[1][0] + 1*element_height , ants[0][1]-5])
                vector_origin = (vector_origin[0] + vector_length, vector_origin[1] - 3)
                
                if False:
                    plt.arrow(vector_origin[0],vector_origin[1], 0, vector_length,color='k',label='N',head_width = box_height)
                    plt.text(vector_origin[0] + 0.75,vector_origin[1] + vector_length, 'U',fontsize = label_fontsize)
                    
                    plt.arrow(vector_origin[0],vector_origin[1], vector_length,0,color='k',label='E',head_width = box_height)
                    plt.text(vector_origin[0] + vector_length,vector_origin[1] + 0.75, 'E',fontsize = label_fontsize)
                else:
                    plt.text(vector_origin[0] - 1.25*vector_length ,vector_origin[1]-2, 'E-W Slope',fontsize = label_fontsize)
                


                l = vector_length

                center = vector_origin
                p1 = (vector_origin[0] - l*numpy.cos(numpy.deg2rad(slope)), vector_origin[1] + l*numpy.sin(numpy.deg2rad(slope)))
                p2 = (vector_origin[0] - l, vector_origin[1])
                
                plt.plot([center[0] , p1[0]], [center[1] , p1[1]],c='k')
                plt.plot([center[0] , p2[0]], [center[1] , p2[1]],c='k')

                ang = AngleAnnotation(vector_origin, p1, p2,
                                        ax=ax1, size=60, text= r"$\mathbf{%i^\circ}$"%(slope) + '        ',text_kw=dict(fontsize=12, weight='bold'),textposition='edge')

                plt.ylim(min(n_vals),max(n_vals) + element_height)
                plt.xlim(min(e_vals),max(e_vals))
                #plt.text(0.01, 0.01, 'Antenna\'s Magnified %ix'%antenna_scale_factor, fontsize=magnify_label_fontsize, transform=ax1.transAxes)

                plane_E, plane_N = numpy.meshgrid(numpy.linspace(min(e_vals),max(e_vals), 100), numpy.linspace(min(n_vals),max(n_vals) + element_height, 100))
                plane_U = plane_equation(plane_E, plane_N)

                # plt.contourf(plane_E, plane_N, plane_U, levels = 20, cmap='Greys',alpha=0.4)#numpy.sort(ants[:,2])

                plt.ylim(min(n_vals) - element_height,max(n_vals) + element_height)
                plt.xlim(min(e_vals),max(e_vals))
                plt.tight_layout()

                fig.savefig('./array_schematic_%s.pdf'%mode, dpi=300, transparent=True)