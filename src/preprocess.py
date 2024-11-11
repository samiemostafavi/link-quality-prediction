import os, sys, gzip, json, sqlite3, random, pickle
import plotly.graph_objects as go
import numpy as np
import pandas as pd
from pathlib import Path
from loguru import logger
import plotly.graph_objects as go

from edaf.core.uplink.preprocess import preprocess_ul
from edaf.core.uplink.analyze_channel import ULChannelAnalyzer
from edaf.core.uplink.analyze_packet import ULPacketAnalyzer
from edaf.core.uplink.analyze_scheduling import ULSchedulingAnalyzer
from plotly.subplots import make_subplots

if not os.getenv('DEBUG'):
    logger.remove()
    logger.add(sys.stdout, level="INFO")

# in case you have offline parquet journey files, you can use this script to decompose delay
# pass the address of a folder in argv with the following structure:
# FOLDER_ADDR/
# -- gnb/
# ---- latseq.*.lseq
# -- ue/
# ---- latseq.*.lseq
# -- upf/
# ---- se_*.json.gz

# create database file by running
# python preprocess.py results/240928_082545_results

# it will result a database.db file inside the given directory

def preprocess_edaf(args):

    folder_path = Path(args.source)
    result_database_file = folder_path / 'database.db'

    # GNB
    gnb_path = folder_path.joinpath("gnb")
    gnb_lseq_file = list(gnb_path.glob("*.lseq"))[0]
    logger.info(f"found gnb lseq file: {gnb_lseq_file}")
    gnb_lseq_file = open(gnb_lseq_file, 'r')
    gnb_lines = gnb_lseq_file.readlines()
    
    # UE
    ue_path = folder_path.joinpath("ue")
    ue_lseq_file = list(ue_path.glob("*.lseq"))[0]
    logger.info(f"found ue lseq file: {ue_lseq_file}")
    ue_lseq_file = open(ue_lseq_file, 'r')
    ue_lines = ue_lseq_file.readlines()

    # NLMT
    nlmt_path = folder_path.joinpath("upf")
    nlmt_file = list(nlmt_path.glob("se_*"))[0]
    if nlmt_file.suffix == '.json':
        with open(nlmt_file, 'r') as file:
            nlmt_records = json.load(file)['oneway_trips']
    elif nlmt_file.suffix == '.gz':
        with gzip.open(nlmt_file, 'rt', encoding='utf-8') as file:
            nlmt_records = json.load(file)['oneway_trips']
    else:
        logger.error(f"NLMT file format not supported: {nlmt_file.suffix}")
    logger.info(f"found nlmt file: {nlmt_file}")

    # Open a connection to the SQLite database
    conn = sqlite3.connect(result_database_file)
    # process the lines
    preprocess_ul(conn, gnb_lines, ue_lines, nlmt_records)
    # Close the connection when done
    conn.close()
    logger.success(f"Tables successfully saved to '{result_database_file}'.")


def find_closest_schedule(failed_ul_schedules, ts_value, hapid_value):
    
    # Filter items by hqpid
    closest_item = None
    closest_index = -1
    min_diff = float('inf')

    for index, item in enumerate(failed_ul_schedules):
        if item.get('sched.cause.hqpid') == hapid_value:
            timestamp = item.get('ue_scheduled_ts')
            if timestamp < ts_value:
                diff = ts_value - timestamp
                if diff < min_diff and diff < 0.05:
                    min_diff = diff
                    closest_item = item
                    closest_index = index
    
    return closest_item, closest_index


def find_the_schedule(all_ul_schedules_df, ue_mac_attempt_0):
    
    # make pandas df from all_ul_schedules
    all_ul_schedules_df['ue_scheduled_ts'] = all_ul_schedules_df['ue_scheduled_ts'].astype(float)
    # find the schedule with ue_scheduled_ts closer than 1ms to the given ts_value
    closest_schedule = all_ul_schedules_df.loc[
        (all_ul_schedules_df['sched.cause.hqpid'] == ue_mac_attempt_0['phy.tx.hqpid']) &
        (all_ul_schedules_df['sched.ue.frametx'] == ue_mac_attempt_0['phy.tx.fm']) &
        (all_ul_schedules_df['sched.ue.slottx'] == ue_mac_attempt_0['phy.tx.sl']) &
        ((all_ul_schedules_df['ue_scheduled_ts'] - ue_mac_attempt_0['phy.tx.timestamp']).abs() < 0.001)  # 1ms
    ]

    if not closest_schedule.empty:
        closest_item = closest_schedule.iloc[0].to_dict()
        closest_item_index = closest_schedule.index[0]
    else:
        closest_item = None
        closest_item_index = None
    
    return closest_item, closest_item_index


def process_failed_rlc_events(all_ul_schedules, failed_ue_rlc_attempts, max_harq_attempts=4):
    all_ul_schedules_df = pd.DataFrame(all_ul_schedules)

    # the key that connects these events is the hqpid
    # A Nack and an RLC retransmission are the start
    # first we iterate over all failed rlc attempts, and find their corresponding mac attempts

    # in the end, we filter out the repeated ones
    # so create a set of schedule ids
    unique_schedule_ids = set()

    failed_harq_groups = []

    # ue_rlc_segments_df: ['txpdu_id', 'rlc.txpdu.M1buf', 'rlc.txpdu.R2buf', 'rlc.txpdu.sn', 'rlc.txpdu.srn', 'rlc.txpdu.so', 'rlc.txpdu.tbs', 'rlc.txpdu.timestamp', 'rlc.txpdu.length', 'rlc.txpdu.leno', 'rlc.txpdu.ENTno', 'rlc.txpdu.retx', 'rlc.txpdu.retxc', 'rlc.report.timestamp', 'rlc.report.num', 'rlc.report.ack', 'rlc.report.tpollex', 'mac.sdu.lcid', 'mac.sdu.tbs', 'mac.sdu.frame', 'mac.sdu.slot', 'mac.sdu.timestamp', 'mac.sdu.length', 'mac.sdu.M2buf', 'rlc.resegment.old_leno', 'rlc.resegment.old_so', 'rlc.resegment.other_seg_leno', 'rlc.resegment.other_seg_so', 'rlc.resegment.pdu_header_len', 'rlc.resegment.pdu_len']
    for ue_rlc_row in failed_ue_rlc_attempts:
        #ue_mac_attempts_df: ['mac_id', 'phy.tx.timestamp', 'phy.tx.Hbuf', 'phy.tx.rvi', 'phy.tx.fm', 'phy.tx.sl', 'phy.tx.nb_rb', 'phy.tx.nb_sym', 'phy.tx.mod_or', 'phy.tx.len', 'phy.tx.rnti', 'phy.tx.hqpid', 'mac.harq.timestamp', 'mac.harq.hqpid', 'mac.harq.rvi', 'mac.harq.len', 'mac.harq.ndi', 'mac.harq.M3buf']
        ue_mac_attempt_0 = ue_rlc_row['ue_mac_attempt_0']
        frame = ue_rlc_row['mac.sdu.frame']
        slot = ue_rlc_row['mac.sdu.slot']
        hqpid = ue_mac_attempt_0['mac.harq.hqpid']
        rvi = ue_mac_attempt_0['phy.tx.real_rvi']

        failed_harq_group = {
            'schedule_id': None,
            'frame' : None,
            'slot' : None,
            'timestamp' : None,
            'total_hqrounds' : 0,
            'hqpid' : hqpid,
            'rlc_failed' : True,
            'related_schedules_list' : []
        }
        
        # find it in all_ul_schedules_df
        rlc_failed_schedule, rfs_idx = find_the_schedule(all_ul_schedules_df, ue_mac_attempt_0)
        if rlc_failed_schedule == None:
            logger.warning(f"No schedule found for failed rlc attempt: {ue_rlc_row}")
            failed_harq_group['timestamp'] = ue_mac_attempt_0['phy.tx.timestamp']
            failed_harq_group['frame'] = ue_mac_attempt_0['phy.tx.fm']
            failed_harq_group['slot'] = ue_mac_attempt_0['phy.tx.sl']
            failed_harq_groups.append(failed_harq_group)
            continue

        # gnb_sched_reports_df: ['sched.ue.rnti', 'sched.ue.frame', 'sched.ue.slot', 'sched.ue.frametx', 'sched.ue.slottx', 'sched.ue.tbs', 'sched.ue.mcs', 'sched.ue.timestamp', 'sched.ue.rbs', 'sched.cause.type', 'sched.cause.frame', 'sched.cause.slot', 'sched.cause.diff', 'sched.cause.timestamp', 'sched.cause.buf', 'sched.cause.sched', 'sched.cause.hqround', 'sched.cause.hqpid']
        related_schedules_list = [rlc_failed_schedule]
        
        if pd.isnull(rlc_failed_schedule['sched.cause.hqround']):
            cur_hqround = 0
            # it means it is the first attempt and it was failed
            failed_harq_group['timestamp'] = rlc_failed_schedule['ue_scheduled_ts']
        else:
            cur_hqround = rlc_failed_schedule['sched.cause.hqround']
            # it means it was the middle attempt and it was failed
            # iterate backwards over failed_ul_schedules from rrfs_idx until we find the attempts with 'sched.cause.hqround'<cur_hqround or pd.isnull('sched.cause.hqround') and the same 'sched.cause.hqpid'
            for idx in range(rfs_idx-1, -1, -1):
                prev_schedule = all_ul_schedules[idx]
                # if it too far back, stop
                if abs(prev_schedule['ue_scheduled_ts'] - ue_mac_attempt_0['phy.tx.timestamp']) > 0.050: # 50ms
                    break
                if pd.isnull(prev_schedule['sched.cause.hqround']) and prev_schedule['sched.cause.hqpid'] == hqpid:
                    related_schedules_list.append(prev_schedule)
                    break
                if prev_schedule['sched.cause.hqpid'] == hqpid and prev_schedule['sched.cause.hqround'] < cur_hqround:
                    related_schedules_list.append(prev_schedule)
        
        # iterate forward over failed_ul_schedules from rrfs_idx until we find the attempts with 'sched.cause.hqround'>cur_hqround and the same 'sched.cause.hqpid'
        for idx in range(rfs_idx+1, len(all_ul_schedules)):
            next_schedule = all_ul_schedules[idx]
            # if it too far, stop
            if abs(next_schedule['ue_scheduled_ts'] - ue_mac_attempt_0['phy.tx.timestamp']) > 0.050: # 50ms
                break
            if next_schedule['sched.cause.hqpid'] == hqpid and next_schedule['sched.cause.hqround'] == (max_harq_attempts-1):
                related_schedules_list.append(next_schedule)
                break
            if next_schedule['sched.cause.hqpid'] == hqpid and next_schedule['sched.cause.hqround'] > cur_hqround:
                related_schedules_list.append(next_schedule)
        
        failed_harq_group['total_hqrounds'] += len(related_schedules_list)
            
        # sort the list based on the timestamp
        related_schedules_list = sorted(related_schedules_list, key=lambda x: x['ue_scheduled_ts'], reverse=False)

        failed_harq_group['timestamp'] = related_schedules_list[0]['ue_scheduled_ts']
        failed_harq_group['frame'] = related_schedules_list[0]['sched.ue.frametx']
        failed_harq_group['slot'] = related_schedules_list[0]['sched.ue.slottx']
        failed_harq_group['schedule_id'] = related_schedules_list[0]['schedule_id']
        failed_harq_group['related_schedules_list'] = related_schedules_list
        if failed_harq_group['schedule_id'] not in unique_schedule_ids:
            failed_harq_groups.append(failed_harq_group)
            unique_schedule_ids.add(failed_harq_group['schedule_id'])

    return failed_harq_groups


def process_successful_retx_schedule_events(all_ul_schedules, successful_retx_schedules):

    all_ul_schedules_df = pd.DataFrame(all_ul_schedules)

    # the key that connects these events is the hqpid
    # A Nack and an RLC retransmission are the start
    # first we iterate over all failed rlc attempts, and find their corresponding mac attempts

    # in the end, we filter out the repeated ones
    # so create a set of schedule ids
    unique_schedule_ids = set()

    failed_harq_groups = []

    for suc_retx_gnb_schedule in successful_retx_schedules:
        # gnb_sched_reports_df: ['sched.ue.rnti', 'sched.ue.frame', 'sched.ue.slot', 'sched.ue.frametx', 'sched.ue.slottx', 'sched.ue.tbs', 'sched.ue.mcs', 'sched.ue.timestamp', 'sched.ue.rbs', 'sched.cause.type', 'sched.cause.frame', 'sched.cause.slot', 'sched.cause.diff', 'sched.cause.timestamp', 'sched.cause.buf', 'sched.cause.sched', 'sched.cause.hqround', 'sched.cause.hqpid']

        related_schedules_list = [suc_retx_gnb_schedule]
        hqpid = suc_retx_gnb_schedule['sched.cause.hqpid']
        hqround = suc_retx_gnb_schedule['sched.cause.hqround']

        failed_harq_group = {
            'frame' : None,
            'slot' : None,
            'timestamp' : None,
            'total_hqrounds' : hqround+1,
            'hqpid' : hqpid,
            'rlc_failed' : False,
            'schedule_id' : None,
            'related_schedules_list' : []
        }

        rfs_idx = all_ul_schedules_df.loc[
            (all_ul_schedules_df['schedule_id'] == suc_retx_gnb_schedule['schedule_id'])
        ].index[0]

        # it means there is still more failed schedules to go through
        # iterate backwards over failed_ul_schedules from rrfs_idx until we find the attempts with 'sched.cause.hqround'<cur_hqround or pd.isnull('sched.cause.hqround') and the same 'sched.cause.hqpid'
        for idx in range(rfs_idx-1, -1, -1):
            failed_schedule = all_ul_schedules[idx]
            # if it too far back, stop
            if abs(failed_schedule['ue_scheduled_ts'] - suc_retx_gnb_schedule['sched.ue.timestamp']) > 0.050: # 50ms
                break
            if pd.isnull(failed_schedule['sched.cause.hqround']) and failed_schedule['sched.cause.hqpid'] == hqpid:
                related_schedules_list.append(failed_schedule)
                break
            if failed_schedule['sched.cause.hqpid'] == hqpid and failed_schedule['sched.cause.hqround'] < hqround:
                related_schedules_list.append(failed_schedule)
           
        # sort the list based on the timestamp
        related_schedules_list = sorted(related_schedules_list, key=lambda x: x['ue_scheduled_ts'], reverse=False)

        failed_harq_group['frame'] = related_schedules_list[0]['sched.ue.frametx']
        failed_harq_group['slot'] = related_schedules_list[0]['sched.ue.slottx']
        failed_harq_group['timestamp'] = related_schedules_list[0]['ue_scheduled_ts']
        failed_harq_group['schedule_id'] = related_schedules_list[0]['schedule_id']
        failed_harq_group['related_schedules_list'] = related_schedules_list

        if failed_harq_group['schedule_id'] not in unique_schedule_ids:
            failed_harq_groups.append(failed_harq_group)
            unique_schedule_ids.add(failed_harq_group['schedule_id'])

    return failed_harq_groups

def filter_successful_retx_schedule(grouped_retx_schedules, grouped_rlc_failed_schedules):

    logger.info(f"Number of successful retx schedules before filtering: {len(grouped_retx_schedules)}")

    results = []
    for retx_group in grouped_retx_schedules:
        found_similar = False
        for rlc_group in grouped_rlc_failed_schedules:
            if (retx_group['hqpid'] == rlc_group['hqpid'] and
                    abs(retx_group['timestamp'] - rlc_group['timestamp']) < 0.001): # 1ms
                logger.info(f"Found similar: {retx_group['frame']} {retx_group['slot']}, {rlc_group['frame']}, {rlc_group['slot']}")
                found_similar = True
                break

        if not found_similar:
            results.append(retx_group)

    logger.info(f"Number of successful retx schedules after filtering: {len(results)}")
    return results


def plot_link_data(args):

    # read configuration from args.config
    with open(args.config, 'r') as f:
        config = json.load(f)
    # select the source configuration
    config = config[args.configname]

    # read experiment configuration
    folder_addr = Path(args.source)
    result_database_file = folder_addr / 'database.db'

    # read exp configuration from args.config
    with open(folder_addr / 'experiment_config.json', 'r') as f:
        exp_config = json.load(f)

    time_mask = config['time_mask']
    filter_packet_sizes = config['filter_packet_sizes']
    window_config = config['window_config']
    dataset_size_max = config['dataset_size_max']
    split_ratios = config['split_ratios']
    dtime_max = config['dtime_max']
    
    slots_duration_ms = exp_config['slots_duration_ms']
    num_slots_per_frame = exp_config['slots_per_frame']
    total_prbs_num = exp_config['total_prbs_num']
    symbols_per_slot = exp_config['symbols_per_slot']
    scheduling_map_num_integers = exp_config['scheduling_map_num_integers']
    max_num_frames = exp_config['max_num_frames']
    scheduling_time_ahead_ms = exp_config['scheduling_time_ahead_ms']
    max_harq_attempts = exp_config['max_harq_attempts']

    # prepare the results folder
    results_folder_addr = folder_addr / 'link_plots' / args.name
    results_folder_addr.mkdir(parents=True, exist_ok=True)
    with open(results_folder_addr / 'config.json', 'w') as f:
        json_obj = json.dumps(config, indent=4)
        f.write(json_obj)

    # initiate the analyzer
    chan_analyzer = ULChannelAnalyzer(result_database_file)
    packet_analyzer = ULPacketAnalyzer(result_database_file)
    sched_analyzer = ULSchedulingAnalyzer(
        total_prbs_num = total_prbs_num, 
        symbols_per_slot = symbols_per_slot,
        slots_per_frame = num_slots_per_frame, 
        slots_duration_ms = slots_duration_ms, 
        scheduling_map_num_integers = scheduling_map_num_integers,
        max_num_frames = max_num_frames,
        db_addr = result_database_file
    )
    experiment_length_ts = chan_analyzer.last_ts - chan_analyzer.first_ts
    logger.info(f"Total experiment duration: {(experiment_length_ts)} seconds")

    begin_ts = chan_analyzer.first_ts+experiment_length_ts*time_mask[0]
    end_ts = chan_analyzer.first_ts+experiment_length_ts*time_mask[1]
    logger.info(f"Filtering link events from {begin_ts} to {end_ts}, duration: {experiment_length_ts*time_mask[1]-experiment_length_ts*time_mask[0]} seconds")

    # find the packet arrivals
    packet_arrivals = packet_analyzer.figure_packet_arrivals_from_ts(begin_ts, end_ts)
    logger.info(f"Number of packet arrivals for this duration: {len(packet_arrivals)}")
    arrivals_ts_list = [(item['ip.in.timestamp']-begin_ts)*1000 for item in packet_arrivals]
    arrivals_size_list = [item['ip.in.length'] for item in packet_arrivals]

    # find the RNTI of the stream
    packets = packet_analyzer.figure_packettx_from_ts(begin_ts, begin_ts+1.0) # just take one second of packets
    packets_rnti_set = set([item['rlc.attempts'][0]['rnti'] for item in packets])
    # remove None from the set
    packets_rnti_set.discard(None)
    logger.info(f"RNTIs in the packet stream: {packets_rnti_set}")
    if len(packets_rnti_set) > 1:
        logger.error("Multiple RNTIs in the packet stream, exiting...")
        return
    stream_rnti = list(packets_rnti_set)[0]

    # extract MCS value time series
    mcs_arr = chan_analyzer.find_mcs_from_ts(begin_ts,end_ts)
    set_rnti = set([item['rnti'] for item in mcs_arr])
    logger.info(f"Number of unique RNTIs in MCS indices: {len(set_rnti)}")
    # filter out the MCS values for the stream RNTI
    mcs_val_list =  np.array([item['mcs'] for item in mcs_arr if item['rnti'] == stream_rnti])
    mcs_ts_list =  np.array([(item['timestamp']-begin_ts)*1000 for item in mcs_arr if item['rnti'] == stream_rnti])


    if args.fast:
        # Create a subplot figure with 2 rows
        fig = make_subplots(rows=2, cols=1, subplot_titles=('MCS Index', 'Packet arrivals'))
        fig.add_trace(go.Scatter(x=mcs_ts_list, y=mcs_val_list, mode='lines+markers', name='MCS value', marker=dict(symbol='circle')), row=1, col=1)
        fig.add_trace(go.Scatter(x=arrivals_ts_list, y=arrivals_size_list, mode='markers', name='Packet arrivals', marker=dict(symbol='square')), row=2, col=1)

        # find repeated RLC attempts
        repeated_ue_rlc_attempts = chan_analyzer.find_repeated_ue_rlc_attempts_from_ts(begin_ts, end_ts)
        repeated_ue_rlc_val_list = np.array([0 for _ in repeated_ue_rlc_attempts])
        repeated_ue_rlc_ts_list = np.array([(item['rlc.txpdu.timestamp']-begin_ts)*1000 for item in repeated_ue_rlc_attempts])

        # find MAC attempts with ndi=0 (NACKs basically)
        ue_ndi0_mac_attempts = chan_analyzer.find_ndi0_ue_mac_attempts_from_ts(begin_ts, end_ts)
        ue_ndi0_mac_val_list = np.array([item['phy.tx.real_rvi'] for item in ue_ndi0_mac_attempts])
        ue_ndi0_mac_text_list = np.array([item['mac.harq.hqpid'] for item in ue_ndi0_mac_attempts])
        ue_ndi0_mac_ts_list = np.array([(item['phy.tx.timestamp']-begin_ts)*1000 for item in ue_ndi0_mac_attempts])

        # for failed_ue_rlc attempts:
        fig.add_trace(go.Scatter(x=repeated_ue_rlc_ts_list, y=repeated_ue_rlc_val_list-0.5, mode='markers', name='Repeated RLC attempts', marker=dict(symbol='triangle-down')), row=1, col=1)
        
        # for ue_ndi0_mac_val_list:
        fig.add_trace(go.Scatter(x=ue_ndi0_mac_ts_list, y=ue_ndi0_mac_val_list-0.3, mode='markers+text', name='Ue mac ndi0', marker=dict(symbol='triangle-up'), text=ue_ndi0_mac_text_list, textposition='top center'), row=1, col=1)

        fig.update_layout(
            title='Link Data Plots',
            xaxis_title='Time [ms]',
            yaxis_title='Values',
            legend_title='Legend',
        )
        fig.update_xaxes(title_text='Time [ms]', row=1, col=1)
        fig.update_yaxes(title_text='Values', row=1, col=1)
        fig.update_xaxes(title_text='Time [ms]', row=2, col=1)
        fig.update_yaxes(title_text='Values', row=2, col=1)
        fig.update_xaxes(matches='x')
        fig.write_html(str(results_folder_addr / 'fast_plot.html'))

    else:
        # find repeated RLC attempts
        failed_ue_rlc_attempts = chan_analyzer.find_failed_ue_rlc_segments_from_ts(begin_ts, end_ts)
        failed_ue_rlc_val_list = np.array([0 for _ in failed_ue_rlc_attempts])
        failed_ue_rlc_ts_list = np.array([(item['rlc.txpdu.timestamp']-begin_ts)*1000 for item in failed_ue_rlc_attempts])

        # find all and failed scheduling events
        all_ul_schedules = sched_analyzer.find_all_schedules_from_ts(begin_ts, end_ts, stream_rnti, scheduling_time_ahead_ms/1000)
        failed_ul_schedules = sched_analyzer.find_failed_schedules_from_ts(begin_ts, end_ts, stream_rnti, scheduling_time_ahead_ms/1000)
        failed_ul_schedules_ts_list = np.array([(item['ue_scheduled_ts']-begin_ts)*1000 for item in failed_ul_schedules])
        failed_ul_schedules_text_list = np.array([ item['sched.cause.hqpid'] for item in failed_ul_schedules])
        failed_ul_schedules_val_list = np.array([ item['sched.cause.hqround'] if item['sched.cause.type'] == 4 else 0 for item in failed_ul_schedules])
        
        # find failed ue mac attempts
        failed_ue_mac_attempts = chan_analyzer.find_failed_ue_mac_attempts_from_ts(begin_ts, end_ts, stream_rnti)
        failed_ue_mac_text_list = np.array([ item['mac.harq.hqpid'] for item in failed_ue_mac_attempts ])
        failed_ue_mac_val_list = np.array([ item['phy.tx.real_rvi'] for item in failed_ue_mac_attempts ])
        failed_ue_mac_ts_list = np.array([ (item['phy.tx.timestamp']-begin_ts)*1000 for item in failed_ue_mac_attempts ])
        
        # find retx schedules
        retx_schedules = sched_analyzer.find_retx_schedules_from_ts(begin_ts, end_ts, stream_rnti, SCHED_OFFSET_S=scheduling_time_ahead_ms/1000)
        successful_retx_schedule = sched_analyzer.find_retx_schedules_from_ts(begin_ts, end_ts, stream_rnti, SCHED_OFFSET_S=scheduling_time_ahead_ms/1000, only_successful=True)
        retx_schedules_val_list = np.array([item['sched.cause.hqround'] for item in retx_schedules])
        retx_schedules_text_list = np.array([item['sched.cause.hqpid'] for item in retx_schedules])
        retx_schedules_ts_list = np.array([(item['ue_scheduled_ts']-begin_ts)*1000 for item in retx_schedules])

        # find MAC attempts with ndi=0 (NACKs basically)
        ue_ndi0_mac_attempts = chan_analyzer.find_ndi0_ue_mac_attempts_from_ts(begin_ts, end_ts)
        ue_ndi0_mac_val_list = np.array([item['phy.tx.real_rvi'] for item in ue_ndi0_mac_attempts])
        ue_ndi0_mac_text_list = np.array([item['mac.harq.hqpid'] for item in ue_ndi0_mac_attempts])
        ue_ndi0_mac_ts_list = np.array([(item['phy.tx.timestamp']-begin_ts)*1000 for item in ue_ndi0_mac_attempts])

        # process RLC retransmissions
        grouped_rlc_failed_schedules = process_failed_rlc_events(all_ul_schedules, failed_ue_rlc_attempts, max_harq_attempts)
        grouped_rlc_failed_schedules_val_list = np.array([item['total_hqrounds'] for item in grouped_rlc_failed_schedules])
        grouped_rlc_failed_schedules_text_list = np.array([item['hqpid'] for item in grouped_rlc_failed_schedules])
        grouped_rlc_failed_schedules_ts_list = np.array([(item['timestamp']-begin_ts)*1000 for item in grouped_rlc_failed_schedules])

        # process successful MAC retransmissions
        grouped_retx_schedules = process_successful_retx_schedule_events(all_ul_schedules, successful_retx_schedule)
        # filter out the successful retx schedules that are actually failed
        grouped_retx_schedules = filter_successful_retx_schedule(grouped_retx_schedules, grouped_rlc_failed_schedules)
        grouped_retx_schedules_val_list = np.array([item['total_hqrounds'] for item in grouped_retx_schedules])
        grouped_retx_schedules_text_list = np.array([item['hqpid'] for item in grouped_retx_schedules])
        grouped_retx_schedules_ts_list = np.array([(item['timestamp']-begin_ts)*1000 for item in grouped_retx_schedules])

        # Create a subplot figure with 2 rows
        fig = make_subplots(rows=3, cols=1, subplot_titles=('MCS Index and packet arrivals', 'Link Quality', 'Processed Events'))
        fig.add_trace(go.Scatter(x=mcs_ts_list, y=mcs_val_list, mode='lines+markers', name='MCS value', marker=dict(symbol='circle')), row=1, col=1)
        fig.add_trace(go.Scatter(x=arrivals_ts_list, y=arrivals_size_list, mode='markers', name='Packet arrivals', marker=dict(symbol='square')), row=1, col=1)

        # for failed_ul_schedules:
        fig.add_trace(go.Scatter(x=failed_ul_schedules_ts_list, y=failed_ul_schedules_val_list, mode='markers+text', name='Failed UL schedules', marker=dict(symbol='square'), text=failed_ul_schedules_text_list, textposition='top center'), row=2, col=1)
        # for failed_ue_mac_attempts:
        fig.add_trace(go.Scatter(x=failed_ue_mac_ts_list, y=failed_ue_mac_val_list-0.1, mode='markers+text', name='Failed UE mac attempts', marker=dict(symbol='circle'), text=failed_ue_mac_text_list, textposition='top center'), row=2, col=1)
        # for retx_schedules_val_list:
        fig.add_trace(go.Scatter(x=retx_schedules_ts_list, y=retx_schedules_val_list-0.2, mode='markers+text', name='Retx schedules', marker=dict(symbol='triangle-up'), text=retx_schedules_text_list, textposition='top center'), row=2, col=1)
        # for ue_ndi0_mac_val_list:
        fig.add_trace(go.Scatter(x=ue_ndi0_mac_ts_list, y=ue_ndi0_mac_val_list-0.3, mode='markers+text', name='Ue mac ndi0', marker=dict(symbol='triangle-up'), text=ue_ndi0_mac_text_list, textposition='top center'), row=2, col=1)
        # for failed_ue_rlc attempts:
        fig.add_trace(go.Scatter(x=failed_ue_rlc_ts_list, y=failed_ue_rlc_val_list-0.5, mode='markers', name='Failed RLC attempts', marker=dict(symbol='triangle-down')), row=2, col=1)
        
        # for grouped_rlc_failed_schedules_val_list:
        fig.add_trace(go.Scatter(x=grouped_rlc_failed_schedules_ts_list, y=grouped_rlc_failed_schedules_val_list, mode='markers+text', name='Processed failed RLC', marker=dict(symbol='triangle-up'), text=grouped_rlc_failed_schedules_text_list, textposition='top center'), row=3, col=1)

        # for grouped_retx_schedules_val_list:
        fig.add_trace(go.Scatter(x=grouped_retx_schedules_ts_list, y=grouped_retx_schedules_val_list+0.1, mode='markers+text', name='Processed retx event', marker=dict(symbol='triangle-down'), text=grouped_retx_schedules_text_list, textposition='top center'), row=3, col=1)

        fig.update_layout(
            title='Link Data Plots',
            xaxis_title='Time [ms]',
            yaxis_title='Values',
            legend_title='Legend',
        )
        fig.update_xaxes(title_text='Time [ms]', row=1, col=1)
        fig.update_yaxes(title_text='Values', row=1, col=1)
        fig.update_xaxes(title_text='Time [ms]', row=2, col=1)
        fig.update_yaxes(title_text='Values', row=2, col=1)
        fig.update_xaxes(title_text='Time [ms]', row=3, col=1)
        fig.update_yaxes(title_text='Values', row=3, col=1)
        fig.update_xaxes(matches='x')
        fig.write_html(str(results_folder_addr / 'combined_plot.html'))

    
def create_training_dataset(args):
    """
    Create a training dataset
    """

    # read configuration from args.config
    with open(args.config, 'r') as f:
        config = json.load(f)
    # select the source configuration
    config = config[args.configname]

    # read experiment configuration
    folder_addr = Path(args.source)
    result_database_file = folder_addr / 'database.db'

    # read exp configuration from args.config
    with open(folder_addr / 'experiment_config.json', 'r') as f:
        exp_config = json.load(f)

    time_mask = config['time_mask']
    filter_packet_sizes = config['filter_packet_sizes']

    # select the source configuration
    window_config = config['window_config']
    dataset_size_max = config['dataset_size_max']
    split_ratios = config['split_ratios']
    dtime_max = config['dtime_max']
    
    slots_duration_ms = exp_config['slots_duration_ms']
    num_slots_per_frame = exp_config['slots_per_frame']
    total_prbs_num = exp_config['total_prbs_num']
    symbols_per_slot = exp_config['symbols_per_slot']
    scheduling_map_num_integers = exp_config['scheduling_map_num_integers']
    max_num_frames = exp_config['max_num_frames']
    scheduling_time_ahead_ms = exp_config['scheduling_time_ahead_ms']
    max_harq_attempts = exp_config['max_harq_attempts']

    # prepare the results folder
    results_folder_addr = folder_addr / 'training_datasets' / args.name
    results_folder_addr.mkdir(parents=True, exist_ok=True)
    with open(results_folder_addr / 'config.json', 'w') as f:
        json_obj = json.dumps(config, indent=4)
        f.write(json_obj)

    # initiate the analyzer
    chan_analyzer = ULChannelAnalyzer(result_database_file)
    packet_analyzer = ULPacketAnalyzer(result_database_file)
    sched_analyzer = ULSchedulingAnalyzer(
        total_prbs_num = total_prbs_num, 
        symbols_per_slot = symbols_per_slot,
        slots_per_frame = num_slots_per_frame, 
        slots_duration_ms = slots_duration_ms, 
        scheduling_map_num_integers = scheduling_map_num_integers,
        max_num_frames = max_num_frames,
        db_addr = result_database_file
    )
    experiment_length_ts = chan_analyzer.last_ts - chan_analyzer.first_ts
    logger.info(f"Total experiment duration: {(experiment_length_ts)} seconds")

    begin_ts = chan_analyzer.first_ts+experiment_length_ts*time_mask[0]
    end_ts = chan_analyzer.first_ts+experiment_length_ts*time_mask[1]
    logger.info(f"Filtering link events from {begin_ts} to {end_ts}, duration: {experiment_length_ts*time_mask[1]-experiment_length_ts*time_mask[0]} seconds")

    # find the RNTI of the stream
    packets = packet_analyzer.figure_packettx_from_ts(begin_ts, begin_ts+1.0) # just take one second of packets
    packets_rnti_set = set([item['rlc.attempts'][0]['rnti'] for item in packets])
    # remove None from the set
    packets_rnti_set.discard(None)
    logger.info(f"RNTIs in the packet stream: {packets_rnti_set}")
    if len(packets_rnti_set) > 1:
        logger.error("Multiple RNTIs in the packet stream, exiting...")
        return
    stream_rnti = list(packets_rnti_set)[0]

    # find successful retx transmissions
    successful_retx_schedule = sched_analyzer.find_retx_schedules_from_ts(begin_ts, end_ts, stream_rnti, SCHED_OFFSET_S=scheduling_time_ahead_ms/1000, only_successful=True)

    # find repeated RLC attempts
    failed_ue_rlc_attempts = chan_analyzer.find_failed_ue_rlc_segments_from_ts(begin_ts, end_ts)

    # find all scheduling events
    all_ul_schedules = sched_analyzer.find_all_schedules_from_ts(begin_ts, end_ts, stream_rnti, scheduling_time_ahead_ms/1000)

    # process RLC retransmissions
    grouped_rlc_failed_schedules = process_failed_rlc_events(all_ul_schedules, failed_ue_rlc_attempts, max_harq_attempts)

    # process successful MAC retransmissions
    grouped_retx_schedules = process_successful_retx_schedule_events(all_ul_schedules, successful_retx_schedule)

    # filter out the retx_schedules that are in the grouped_rlc_failed_schedules
    grouped_retx_schedules = filter_successful_retx_schedule(grouped_retx_schedules, grouped_rlc_failed_schedules)

    # combine the two lists
    combined_events = [ *grouped_retx_schedules, *grouped_rlc_failed_schedules ]
    # sort the events based on timestamp
    combined_events = sorted(combined_events, key=lambda x: x['timestamp'], reverse=False)

    # create prefinal list of events
    # event types: 
    # we have 4 rounds of retransmission: {0,1,2,3}
    # we have successful or unsuccessful RLC segment {0,1}
    dim_process = 8
    # we use (total_hqrounds-1)+(rlc_failed*4) to map the event types to a unique number between 0 and 7
    # in total it is 8 types of events
    prefinal_events_list = []
    for item in combined_events:
        mcs_reports = chan_analyzer.find_mcs_from_ts(item['timestamp']-0.100, item['timestamp'])
        if len(mcs_reports) == 0:
            continue
        mcs_index = mcs_reports[-1]['mcs']
        prefinal_events_list.append({
            'type_event' : int((item['total_hqrounds']-1)+(int(item['rlc_failed'])*4)),
            'timestamp' : item['timestamp'],
            'mcs_index' : mcs_index
        })

    # sort the events based on timestamp
    prefinal_events_list = sorted(prefinal_events_list, key=lambda x: x['timestamp'], reverse=False)

    link_retransmission_events = []
    last_event_ts = 0
    for item in prefinal_events_list:
        frame_start_ts, frame_num, slot_num = sched_analyzer.find_frame_slot_from_ts(
            timestamp=item['timestamp'],
            SCHED_OFFSET_S=scheduling_time_ahead_ms/1000 # 4ms which is 8*slot_duration_ms
        )

        time_since_frame0 = frame_num*num_slots_per_frame*slots_duration_ms + slot_num*slots_duration_ms
        time_since_last_event = time_since_frame0-last_event_ts
        #if time_since_last_event < 0:
        #    time_since_last_event = time_since_frame0 + max_num_frames*num_slots_per_frame*slots_duration_ms
        if time_since_last_event < 0:
            time_since_last_event = time_since_frame0

        last_event_ts = time_since_frame0

        if time_since_last_event > dtime_max:
            continue

        link_retransmission_events.append(
            {
                'type_event' : item['type_event'],
                'time_since_start' : time_since_frame0,
                'time_since_last_event' : time_since_last_event,
                'timestamp' : item['timestamp'],
                'mcs_index' : item['mcs_index']
            }
        )

    if window_config['type'] == 'time':
        dataset = create_training_dataset_time_window(link_retransmission_events, config)
    elif window_config['type'] == 'event':
        dataset = create_training_dataset_event_window(link_retransmission_events, config)
    else:
        logger.error("Invalid window type")

    # shuffle the dataset
    random.shuffle(dataset)

    # print length of dataset
    logger.info(f"Number of total entries in dataset: {len(dataset)}")
    print(dataset[0])

    # split
    train_num = int(len(dataset)*split_ratios[0])
    dev_num = int(len(dataset)*split_ratios[1])
    print("train: ", train_num, " - dev: ", dev_num)
    # train
    train_ds = {
        'dim_process' : int(dim_process),
        'train' : dataset[0:train_num],
    }

    # Save the dictionary to a pickle file
    with open(results_folder_addr / 'train.pkl', 'wb') as f:
        pickle.dump(train_ds, f)
    # dev
    dev_ds = {
        'dim_process' : dim_process,
        'dev' : dataset[train_num:train_num+dev_num],
    }
    # Save the dictionary to a pickle file
    with open(results_folder_addr / 'dev.pkl', 'wb') as f:
        pickle.dump(dev_ds, f)
    # test
    test_ds = {
        'dim_process' : dim_process,
        'test' : dataset[train_num+dev_num:-1],
    }
    # Save the dictionary to a pickle file
    with open(results_folder_addr / 'test.pkl', 'wb') as f:
        pickle.dump(test_ds, f)

    

def create_training_dataset_event_window(link_retransmission_events, config):
    
    # select the source configuration
    window_config = config['window_config']
    history_window_size = window_config['size']
    dataset_size_max = config['dataset_size_max']

    dataset = []
    for idx,_ in enumerate(link_retransmission_events):
        if idx+history_window_size >= len(link_retransmission_events):
            break
        events_window = []
        for pos, event in enumerate(link_retransmission_events[idx:idx+history_window_size]):
            idx_event = pos
            events_window.append(
                {
                    'idx_event' : pos,
                    'type_event': event['type_event'],
                    'time_since_start' : event['time_since_start'],
                    'time_since_last_event' : event['time_since_last_event'],
                    'mcs_index' : event['mcs_index']
                }
            )        

        #print(events)
        dataset.append(events_window)
        if len(dataset) > dataset_size_max:
            break

    return dataset


def create_training_dataset_time_window(link_retransmission_events, config):

    # select the source configuration
    window_config = config['window_config']
    history_window_size = window_config['size']
    dataset_size_max = config['dataset_size_max']

    stop = False
    dataset = []
    for first_event_idx, first_event in enumerate(link_retransmission_events):
        events_window = []
        pos = 0
        for next_event_idx, next_event in enumerate(link_retransmission_events[first_event_idx:]):
            if (next_event['timestamp']-first_event['timestamp']) > history_window_size:
                break
            events_window.append(
                {
                    'idx_event' : pos,
                    'type_event': next_event['type_event'],
                    'time_since_start' : next_event['time_since_start'],
                    'time_since_last_event' : next_event['time_since_last_event'],
                    'mcs_index' : next_event['mcs_index']
                }
            )
            pos += 1
            if next_event_idx+first_event_idx >= len(link_retransmission_events)-1:
                stop = True
                break
        dataset.append(events_window)
        if (len(dataset) > dataset_size_max) or stop:
            break

    return dataset
