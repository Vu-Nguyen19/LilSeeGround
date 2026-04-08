#!/usr/bin/env python
# Downloads ScanNet public data release filtered by a JSON file containing scene_ids
# Run with ./download-scannet-filtered.py -o [out_dir] -f [json_file]
# -*- coding: utf-8 -*-
import argparse
import os
import urllib.request 
import tempfile
import json
import ssl 

ssl._create_default_https_context = ssl._create_unverified_context

BASE_URL = 'http://kaldir.vc.cit.tum.de/scannet/'
TOS_URL = BASE_URL + 'ScanNet_TOS.pdf'
FILETYPES = ['.aggregation.json', '.sens', '.txt', '_vh_clean.ply', '_vh_clean_2.0.010000.segs.json', '_vh_clean_2.ply', '_vh_clean.segs.json', '_vh_clean.aggregation.json', '_vh_clean_2.labels.ply', '_2d-instance.zip', '_2d-instance-filt.zip', '_2d-label.zip', '_2d-label-filt.zip']
FILETYPES_TEST = ['.sens', '.txt', '_vh_clean.ply', '_vh_clean_2.ply']
RELEASES = ['v2/scans', 'v1/scans']
RELEASES_NAMES = ['v2', 'v1']
RELEASE = RELEASES[0]
RELEASE_NAME = RELEASES_NAMES[0]
V1_IDX = 1


def get_release_scans(release_file):
    scan_lines = urllib.request.urlopen(release_file)
    scans = []
    for scan_line in scan_lines:
        scan_id = scan_line.decode('utf8').rstrip('\n')
        scans.append(scan_id)
    return scans


def download_file(url, out_file):
    out_dir = os.path.dirname(out_file)
    if not os.path.isdir(out_dir):
        os.makedirs(out_dir)
    if not os.path.isfile(out_file):
        print('\t' + url + ' > ' + out_file)
        fh, out_file_tmp = tempfile.mkstemp(dir=out_dir)
        f = os.fdopen(fh, 'w')
        f.close()
        urllib.request.urlretrieve(url, out_file_tmp)
        os.rename(out_file_tmp, out_file)
    else:
        print('WARNING: skipping download of existing file ' + out_file)

def download_scan(scan_id, out_dir, file_types, use_v1_sens, skip_existing=False):
    print('Downloading ScanNet ' + RELEASE_NAME + ' scan ' + scan_id + ' ...')
    if not os.path.isdir(out_dir):
        os.makedirs(out_dir)
    for ft in file_types:
        v1_sens = use_v1_sens and ft == '.sens'
        url = BASE_URL + RELEASE + '/' + scan_id + '/' + scan_id + ft if not v1_sens else BASE_URL + RELEASES[V1_IDX] + '/' + scan_id + '/' + scan_id + ft
        out_file = out_dir + '/' + scan_id + ft
        if skip_existing and os.path.isfile(out_file):
            continue
        download_file(url, out_file)
    print('Downloaded scan ' + scan_id)


def main():
    parser = argparse.ArgumentParser(description='Downloads ScanNet public data release filtered by a JSON file.')
    parser.add_argument('-o', '--out_dir', required=True, help='directory in which to download')
    parser.add_argument('-f', '--filter_file', required=True, help='JSON file containing scene_ids')
    parser.add_argument('--v1', action='store_true', help='download ScanNet v1 instead of v2')
    parser.add_argument('--type', help='specific file type to download (.aggregation.json, .sens, .txt, _vh_clean.ply, _vh_clean_2.0.010000.segs.json, _vh_clean_2.ply, _vh_clean.segs.json, _vh_clean.aggregation.json, _vh_clean_2.labels.ply, _2d-instance.zip, _2d-instance-filt.zip, _2d-label.zip, _2d-label-filt.zip)')
    parser.add_argument('--skip_existing', action='store_true', help='skip download of existing files')
    args = parser.parse_args()

    print('By pressing any key to continue you confirm that you have agreed to the ScanNet terms of use as described at:')
    print(TOS_URL)
    print('***')
    print('Press any key to continue, or CTRL-C to exit.')
    key = input('')

    if args.v1:
        global RELEASE
        global RELEASE_NAME
        RELEASE = RELEASES[V1_IDX]
        RELEASE_NAME = RELEASES_NAMES[V1_IDX]

    # Load filtered scene_ids
    with open(args.filter_file, 'r') as f:
        data = json.load(f)
    
    # scene_id in the file are repeating, use a set to avoid duplicate download
    filtered_scene_ids = set()
    for item in data:
        if 'scene_id' in item:
            filtered_scene_ids.add(item['scene_id'])
    
    print(f'Found {len(filtered_scene_ids)} unique scene_ids in {args.filter_file}')

    release_file = BASE_URL + RELEASE + '.txt'
    all_release_scans = get_release_scans(release_file)
    
    file_types = FILETYPES
    release_test_file = BASE_URL + RELEASE + '_test.txt'
    all_release_test_scans = [] if args.v1 else get_release_scans(release_test_file)
    file_types_test = FILETYPES_TEST
    
    out_dir_scans = os.path.join(args.out_dir, 'scans')
    out_dir_test_scans = os.path.join(args.out_dir, 'scans_test')

    if args.type:
        if args.type not in FILETYPES:
            print('ERROR: Invalid file type: ' + args.type)
            return
        file_types = [args.type]
        if args.type in FILETYPES_TEST:
            file_types_test = [args.type]
        else:
            file_types_test = []

    for scan_id in sorted(list(filtered_scene_ids)):
        is_test_scan = scan_id in all_release_test_scans
        if scan_id not in all_release_scans and (not is_test_scan or args.v1):
            print('WARNING: Invalid scan id: ' + scan_id + '. Skipping...')
            continue
        
        out_dir = os.path.join(out_dir_scans, scan_id) if not is_test_scan else os.path.join(out_dir_test_scans, scan_id)
        scan_file_types = file_types if not is_test_scan else file_types_test
        use_v1_sens = not is_test_scan
        
        download_scan(scan_id, out_dir, scan_file_types, use_v1_sens, skip_existing=args.skip_existing)

    print('Done.')


if __name__ == "__main__": 
    main()
