import numpy as np
import spiceypy as spice
import os
from ..paths import irdir

def parse_ir_line(line):
    #parse the file so that the time and other info are seperate

    #first three fields have no spaces, so split by whitespace
    split=line.split()

    sclk=split[0]
    et=split[1]
    type0=split[2]
    type1=split[3]
    
    #put all the info from the last field back together
    type2=' '.join(split[4:])

    return {'sclk':sclk,
            'et':et,
            'type0':type0,
            'type1':type1,
            'type2':type2}



obsid_dict={0:'periapse',
            1:'outcorona',
            2:'apoapse',
            3:'inspace',
            4:'echelle outcorona',
            5:'echelle inspace',
            6:'occultation',#'stellar, main slit',
            7:'occultation',#'stellar, small keyhole',
            8:'occultation',#'stellar, big keyhole',
            9:'outlimb',
            10:'outdisk',
            11:'echelle outlimb',
            12:'echelle outdisk',
            13:'echelle apoapse',
            14:'echelle periapse',
            15:'centroid',
            16:'incorona',
            17:'inlimb',
            18:'indisk',
            19:'outspace',
            20:'echelle incorona',
            21:'echelle inlimb',
            22:'echelle indisk',
            23:'echelle outspace',
            24:'star',
            25:'echelle relay',
            26:'relay',
            27:'echelle comm',
            28:'comm'
           }





def parse_obsid(full_obsid_int,obs_et):
    #see Chris's iuvs_obsid in the level1a folder for some more info
    #obs_ids are a bitstring. Originally this was intended to have only four bits, but a fifth was needed later on.
    #first four bits in string define old four-bit obs-id
    #fifth+sixth? bit defines whether a nonlinear binning table is used
    #originally, the remaining bits were intended to identify the binning table.
    #However, because of the need for an extra bit for the obs id, the sixth bit is actually now the most significant bit of the obs id

    #technically the table ID could be used to determine which files have hifi in their filenames but I don't want to parse table IDs so we'll handle this another way

    #obsid were reported as hex before 2014 Nov 30
    if obs_et < spice.str2et('2014-Nov-30 00:00:00 UTC'):
        full_obsid_int=int(str(full_obsid_int),16)

    #get the bitstring
    obsid_bitstring = '{0:016b}'.format(int(full_obsid_int))

    #now convert back to base-10 for the dictionary lookup
    obsid_phase = int(obsid_bitstring[6]+obsid_bitstring[:4],2)
    nonlinear   = obsid_bitstring[5]
    table       = int(obsid_bitstring[7:])
    
    phase = obsid_dict[obsid_phase]
    
    #correct for observations of phobos
    if table == 24:
        phase = 'phobos'

    return phase





def construct_integrated_orbit(ir_list,orbit_start=None):
    #concatenates all of the IR files, sorts, deletes duplicate entries, and returns the orbit subset
    #orbit start is a date/time string with the same format as an IUVS filename, e.g. 20141110T145624
    
    #we need to select only files with the same version number to avoid duplicate spacecraft clock lines
    ir_version = [int(ir.split("_v")[1][:2].replace(".","")) for ir in ir_list]
    ir_version_max = np.max(ir_version)
    ir_list = [ir for ir,version in zip(ir_list,ir_version) if version==ir_version_max]
    
    #if there's only one file left look for the ones before and after
    
    
    all_ir_lines=[]
    for ir in ir_list:
        #print(ir)

        #read the file and discard the header and footer
        with open(irdir+ir, 'r') as irfile:
            ir_text = irfile.read()
        ir_text=ir_text.split('\n')
        
        #some files have a header, others don't
        if ir_text[0][0:4]=='File':
            ir_firstline = [i+1 for i,irline in enumerate(ir_text) if irline[0:3]=="---"][0]
            ir_lastline = [i+ir_firstline for i,irline in enumerate(ir_text[ir_firstline:]) if irline==""][0]
        else:
            ir_firstline = 0
            ir_lastline = len(ir_text)-1

        ir_text=ir_text[ir_firstline:ir_lastline]
        
        all_ir_lines.append(ir_text)
    
    all_ir_lines=np.concatenate(all_ir_lines)
    #remove duplicate lines
    #because the ETs can vary due to difference in the spacecraft clock kernel in use when the report was created, we do this based on the SCLK and command
    ir_text_parsed=list(map(parse_ir_line,all_ir_lines))
    ir_text_parsed=np.array(ir_text_parsed)
    ir_unique_identifier=[" ".join([parsed_line['sclk'],parsed_line['type0'],parsed_line['type1'],parsed_line['type2']]) for parsed_line in ir_text_parsed]
    
    import pandas as pd
    unique_indices=list(pd.Series(ir_unique_identifier).drop_duplicates().index)
    ir_text_parsed=ir_text_parsed[unique_indices]
    
    for i,parsed_line in enumerate(ir_text_parsed):
        #correct the et based on the most recent spacecraft clock kernel:
        ir_text_parsed[i]['et']=spice.et2utc(spice.scs2e(spice.bodn2c('MAVEN'),parsed_line['sclk']),'ISOD',3)
    
    #now sort by spacecraft clock
    ir_text_parsed=ir_text_parsed[np.argsort([float(a['sclk']) for a in ir_text_parsed])]
    
    if orbit_start!=None:
        #look for the ET corresponding to the date/time string passed
        spice_parseable_datetime='20'+orbit_start[:2]+'-'+orbit_start[2:4]+'-'+orbit_start[4:6]+'T'+orbit_start[7:9]+':'+orbit_start[9:11]+':'+orbit_start[11:13]
        file_start_et=spice.str2et(spice_parseable_datetime)
        #print(file_start_et)

        ir_periapse_segment_set_line = [i for i,irline in enumerate(ir_text_parsed) if 'vm_string=PERIAPSE' in irline['type2']]
        ir_periapse_segment_set_et = [spice.str2et(ir_text_parsed[i]['et']) for i in ir_periapse_segment_set_line]
        #print(ir_periapse_segment_set_et)

        ir_which_periapse_start=np.searchsorted(ir_periapse_segment_set_et,file_start_et)-1
        #print(ir_which_periapse_start)
        #print(ir_periapse_segment_set_et[ir_which_periapse_start])

        #get the orbit start and end time using the start of the periapse segment
        ir_orbit_start_line=int(ir_periapse_segment_set_line[ir_which_periapse_start])
        ir_orbit_end_line=int(ir_periapse_segment_set_line[ir_which_periapse_start+1])
        #print(ir_orbit_start_line)
        #print(ir_orbit_end_line)

        #restrict the search for all other events to this orbit only
        ir_text_parsed=ir_text_parsed[ir_orbit_start_line:ir_orbit_end_line]
    
    #print(ir_text_parsed)
    
    return ir_text_parsed







def get_integrated_report_info(orbit_start, orbit_end, orbno):
    #orbit_date_start and _end are the datetime portion of the filename for the first and last file in IUVS orbit orbno 
    #these are used to look up the correct integrated report
    
    #first thing we need to do is figure out which integrated report file to use    
    #there is usually more than one integrated report covering the orbit
    irlist=sorted(os.listdir(irdir))
    orbit_date_start=orbit_start[:6]
    orbit_date_end=orbit_end[:6]
    #print(orbit_date_start," - ", orbit_date_end)
    ir_possible=[irname for irname in irlist if ( ((int(irname.split("_")[2][-6:])<=int(orbit_date_start)
                                                    and int(orbit_date_end)<=int(irname.split("_")[3][-6:])) #date range of files contained in filename
                                                   or int(orbit_date_start)==int(irname.split("_")[3][-6:])   #first day in orbit name is last day of filename
                                                   or int(orbit_date_start)-1==int(irname.split("_")[3][-6:]) #day before first day in orbit name is last day of filename
                                                   or int(irname.split("_")[2][-6:])==int(orbit_date_end)     #last day of orbit name is first day of filename 
                                                   or int(irname.split("_")[2][-6:])==int(orbit_date_end)+1   #last day of orbit name is first day of filename 
                                                  )
                                                  and os.stat(irdir+irname).st_size>0)]
    #print(ir_possible)
    
    ir_text_parsed = construct_integrated_orbit(ir_possible,orbit_start)
    
    #print(ir_text_parsed)

    #let's get the times from the file
    ir_orbit_start_time=ir_text_parsed[0]['et']
    ir_orbit_start_et=spice.str2et(ir_orbit_start_time)
    
    ir_orbit_end_time=ir_text_parsed[-1]['et']
    ir_orbit_end_et=spice.str2et(ir_orbit_end_time)
    
    #define the time at the midpoint of the periapsis segment and the apoapsis segment
    ir_outbound_start_et = [spice.str2et(irline['et']) for irline in ir_text_parsed if 'vm_string=OB_SIDE' in irline['type2']][0]
    ir_mid_periapse_et = (ir_orbit_start_et+ir_outbound_start_et)/2
    
    ir_apoapse_start_et = [spice.str2et(irline['et']) for irline in ir_text_parsed if 'vm_string=APOAPSE' in irline['type2']][0]
    ir_inbound_start_et = [spice.str2et(irline['et']) for irline in ir_text_parsed if 'vm_string=IB_SIDE' in irline['type2']][0]
    ir_mid_apoapse_et = (ir_apoapse_start_et+ir_inbound_start_et)/2

    orbit_segment    = np.array(['periapse',        'outbound',           'apoapse',           'inbound'          ])
    orbit_segment_et = np.array([ir_orbit_start_et, ir_outbound_start_et, ir_apoapse_start_et, ir_inbound_start_et])
    
    #now let's look for IUVS image_init commands

    #obs_id is set before a group of images:
    obsid_commands=[(irline['et'],irline['type2'].split(',rsdpu_obs_id=')[1].split(",")[0]) for irline in ir_text_parsed if 'RSP_OBS_ID_SET' in irline['type1']]
    #print(obsid_commands)
    obsid_tag_et,obsid_tag=np.transpose([(spice.str2et(et),parse_obsid(obsid,spice.str2et(et))) for et,obsid in obsid_commands])

    #mcp_level operates in the same way to distinguish light from dark images
    mcp_level_set_commands=[(irline['et'],irline['type2'].split(',rsdpu_level=')[1].split(",")[0]) for irline in ir_text_parsed if ('RSP_MCP_LVL_SET' in irline['type1'] 
                                                                                                                                    and ('rsdpu_detector=FUV' in irline['type2'] 
                                                                                                                                         or 'rsdpu_detector=BOTH' in irline['type2']))]
    mcp_dark_et, mcp_dark=np.transpose([(spice.str2et(et),'dark' if mcp_level=="0x0000" else "light") for et,mcp_level in mcp_level_set_commands])

    #we need to get integration number to check for outdisk and outlimb on orbits < 1050
    img_num_set_commands=[(irline['et'],irline['type2'].split(',rsdpu_img_num=')[1].split(",")[0]) for irline in ir_text_parsed if 'RSP_IMG_NUM_SET' in irline['type1']]
    img_num_et, img_num=np.transpose([(spice.str2et(et),num) for et,num in img_num_set_commands])
    
    #get the image cadence also so we can calculate the length of this observation
    img_cad_set_commands=[(irline['et'],irline['type2'].split(',rsdpu_img_cadence=')[1].split(",")[0]) for irline in ir_text_parsed if 'RSP_IMG_CAND_SET' in irline['type1']]
    img_cad_et, img_cad=np.transpose([(spice.str2et(et),num) for et,num in img_cad_set_commands])    
    

    
    #all of the image_inits have only an et
    img_init_commands_et=[spice.str2et(irline['et']) for irline in ir_text_parsed if 'RSP_IMAGING_INIT' in irline['type1']]

    #print a list of all the commands we've looked at to see the timing
    #all_commands_et  =np.concatenate([obsid_tag_et,mcp_dark_et,img_num_et,img_init_commands_et                  ])
    #all_commands_text=np.concatenate([obsid_tag   ,mcp_dark   ,img_num   ,["init" for i in img_init_commands_et]])
    #all_commands_order=np.argsort(all_commands_et)
    #[print(all_commands_et[i],":",all_commands_text[i]) for i in all_commands_order]
    

    
    #but we can find the most recent obs_id and mcp_level commands that happened before them
    img_init_segment = orbit_segment[np.searchsorted(orbit_segment_et    , img_init_commands_et)-1]
    img_init_obsid   = obsid_tag    [np.searchsorted(obsid_tag_et        , img_init_commands_et)-1]
    img_init_dark    = mcp_dark     [np.searchsorted(mcp_dark_et         , img_init_commands_et)-1]
    img_init_num     = img_num      [np.searchsorted(img_num_et          , img_init_commands_et)-1]
    img_init_num     = np.array(list(map(int  ,img_init_num)))
    img_init_cad     = img_cad      [np.searchsorted(img_cad_et          , img_init_commands_et)-1]
    img_init_cad     = np.array(list(map(float,img_init_cad)))
    #[print(et,": (",spice.et2utc(et,"C",0),")",img_init_segment[i],", ",img_init_obsid[i],", ", img_init_dark[i],", ",img_init_num[i]," @ ",img_init_cad[i]," ms") for i,et in enumerate(img_init_commands_et)]
    
    #now we can generate a list of expected files to be generated
    img_fileinfo=[{'et_start':et_start,
                   'segment':segment,
                   'obsid':obsid.replace('echelle ',''),
                   'echelle':True if 'echelle' in obsid else False,
                   'n_int':num,
                   'et_end':et_start+num*cad/1000.} for et_start,segment,obsid,dark,num,cad in zip(img_init_commands_et,
                                                                                                   img_init_segment,
                                                                                                   img_init_obsid,
                                                                                                   img_init_dark,
                                                                                                   img_init_num,
                                                                                                   img_init_cad) if dark!='dark']
    #[print(img) for img in img_fileinfo]
    
    #do some small corrections on the files
    for i, img in enumerate(img_fileinfo):
        if not img['echelle']:
            if img['obsid']=='periapse':
                #some periapse files are really periapse hifi
                if img['n_int']==1:
                    img_fileinfo[i]['obsid']='periapsehifi'
            if orbno < 1050:
                #outbound files before this orbit were not distinguished by obsid
                if img['obsid']=='outcorona':
                    if img['n_int']==72 or img['n_int']==88 or img['n_int']==92:
                        img_fileinfo[i]['obsid']='outdisk'
                    elif img['n_int']==60:
                        img_fileinfo[i]['obsid']='outlimb'

    return {'orbit_start_et':ir_orbit_start_et,
            'orbit_end_et':ir_orbit_end_et,
            'mid_peri_et':ir_mid_periapse_et,
            'mid_apo_et':ir_mid_apoapse_et,
            'img_list':img_fileinfo}
