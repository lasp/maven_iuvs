#list of the obsids we're trying to put on various frames
plot_obsids={'periapse',
             'outdisk','outlimb','outcorona','outspace',
             'apoapse',
             'inspace','incorona','inlimb','indisk'}

axis_color='#444444'
axis_label_color='#888888'
frame_line_width=0.25
fontsize=4
ticklength=1
tickpad=0.5

orbit_annotation_color='#444444'
orbit_annotation_warning='#884444'
orbit_annotation_fontsize=3
orbit_annotation_linewidth=0.35

pcolormesh_edge_width = 0.05 #to ensure continuous pcolormesh output in saved pdfs

default_target_altitude=135 #altitude to target for orbit annotation arrows

def style_axis(ax,color=axis_color,axis_label_color=axis_label_color):
    #make background a transparent dark gray
    ax.set_facecolor("#222222")
    ax.patch.set_alpha(0)
    
    #set default style
    ax.spines['bottom'].set_color(color)
    ax.spines['bottom'].set_linewidth(frame_line_width)
    ax.spines['top'].set_color(color) 
    ax.spines['top'].set_linewidth(frame_line_width)
    ax.spines['right'].set_color(color)
    ax.spines['right'].set_linewidth(frame_line_width)
    ax.spines['left'].set_color(color)
    ax.spines['left'].set_linewidth(frame_line_width)
    ax.tick_params(axis='both', colors=axis_label_color, which='both', width=frame_line_width, labelsize=fontsize, length=ticklength, pad=tickpad)
    ax.yaxis.label.set_color(axis_label_color)
    ax.xaxis.label.set_color(axis_label_color)
    ax.title.set_color(axis_label_color)
    
    #suppress ticks
    ax.xaxis.set_ticks([])
    ax.yaxis.set_ticks([])

