import os

from lib.test.evaluation.environment import EnvSettings


def local_env_settings():
    settings = EnvSettings()

    # Set your local paths here

    settings.davis_dir = ''
    settings.got10k_path = os.path.expanduser('~') + '/track/data/GOT10k'
    settings.got_packed_results_path = ''
    settings.got_reports_path = ''
    settings.itb_path = os.path.expanduser('~') + '/track/data/ITB'
    settings.lasot_path = os.path.expanduser('~') + '/track/data/LaSOT'
    settings.network_path = os.path.expanduser(
        '~') + '/track/code/GRM/output/test/networks'  # Where tracking networks are stored
    settings.nfs_path = os.path.expanduser('~') + '/track/data/NFS30'
    settings.otb_path = os.path.expanduser('~') + '/track/data/OTB100'
    settings.prj_dir = os.path.expanduser('~') + '/track/code/GRM'
    settings.result_plot_path = os.path.expanduser('~') + '/track/code/GRM/output/test/result_plots'
    settings.results_path = os.path.expanduser(
        '~') + '/track/code/GRM/output/test/tracking_results'  # Where to store tracking results
    settings.save_dir = os.path.expanduser('~') + '/track/code/GRM/output'
    settings.segmentation_path = os.path.expanduser('~') + '/track/code/GRM/output/test/segmentation_results'
    settings.tc128_path = ''
    settings.tn_packed_results_path = ''
    settings.tnl2k_path = ''
    settings.tpl_path = ''
    settings.trackingnet_path = os.path.expanduser('~') + '/track/data/TrackingNet'
    settings.uav_path = os.path.expanduser('~') + '/track/data/UAV123'
    settings.vot18_path = ''
    settings.vot22_path = ''
    settings.vot_path = ''
    settings.avist_path = os.path.expanduser('~') + '/track/data/AVisT'
    settings.youtubevos_dir = ''
    settings.show_result = False
    return settings
