#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 24 15:09:31 2021

@author: pchandrasekar
"""
import json
def fileManipulation():
    with open("data/s3_access.log") as f:
        for line in f:
            logArray = line.split(" ")
        
        print("InputText",inputtext)
        print("LogArray",logArray)
            
    logArray[2] = logArray[2].replace("[", "")
    logArray[3] = logArray[3].replace("]", "")
    logArray[2] = logArray[2] + " "+logArray[3]
    del logArray[3]
    logArray[8] = logArray[8] +" "+logArray[9]+" "+logArray[10]
    del logArray[9:11]
    logArray[8] = logArray[8].replace('"', "")
    logArray[16] = logArray[16]+ " "+logArray[17]+ " "+logArray[18]+ " "+logArray[19]
    del logArray[17:20]
    logArray[15] = logArray[15].replace('"', "")
    logArray[16] = logArray[16].replace('"', "")
    #print(logArray[2] +" "+logArray[3])
    #print(logArray[16],logArray[17])
    print(logArray)
"""    
Expected Values: {'bucket_owner': 'b5c3c9a2b9e7f8c9c508442a00deb47783d0fd203e0c017f10834561f2607f6a', 'bucket_name': 's3testbucketharish', 'request_time': '04/Feb/2021:17:53:36 +0000', 'remote_ip': '98.33.33.216', 'object_size': '-', 'request_id': '647D4303868FF214', 'operation': 'REST.DELETE.OBJECT', 'key': 'harishpalin23.txt', 'request_uri': '"DELETE /harishpalin23.txt HTTP/1.1"', 'http_status': '400', 'error_code': 'AuthorizationHeaderMalformed', 'bytes_sent': '365', 'referrer': '-', 'total_time': '3', 'requester': '-', 'turn_around_time': '"-"', 'user_agent': '"Boto3/1.9.91 Python/3.8.5 Linux/5.4.0-65-generic Botocore/1.12.253"', 'version_id': '-', 'host_id': 'hGyfrBiMikmv9/J1LFIecin4k/B8kBK7i6+FQ0EfNyH3lC8+984oIQja6XOlJtHi9vISwIrfbFU=', 'signature_version': 'SigV4', 'cipher_suite': 'ECDHE-RSA-AES128-GCM-SHA256', 'authentication_type': 'AuthHeader', 'host_header': 's3testbucketharish.s3.amazonaws.com', 'tls_version': 'TLSv1.2', 'source': 's3://premerge-16937950/s3_al_1/1626821762_16937950_premerge_s3_access.log', 'sourcetype': 'aws:s3:accesslogs'}
Actual Values: OrderedDict([('_bkt', 'cdce2etestidx~28~27978B33-D748-44C3-A2F9-EEB611B4190D'), ('_cd', '28:203414'), ('_indextime', '1626822486'), ('_kv', '1'), ('_raw', 'b5c3c9a2b9e7f8c9c508442a00deb47783d0fd203e0c017f10834561f2607f6a s3testbucketharish [04/Feb/2021:17:53:36 +0000] 98.33.33.216 - 647D4303868FF214 REST.DELETE.OBJECT harishpalin23.txt "DELETE /harishpalin23.txt HTTP/1.1" 400 AuthorizationHeaderMalformed 365 - 3 - "-" "Boto3/1.9.91 Python/3.8.5 Linux/5.4.0-65-generic Botocore/1.12.253" - hGyfrBiMikmv9/J1LFIecin4k/B8kBK7i6+FQ0EfNyH3lC8+984oIQja6XOlJtHi9vISwIrfbFU= SigV4 ECDHE-RSA-AES128-GCM-SHA256 AuthHeader s3testbucketharish.s3.amazonaws.com TLSv1.2'), ('_serial', '0'), ('_si', ['idx-i-0972f413e683fd13a.valuable-vulture-6c0.stg.splunkcloud.com', 'cdce2etestidx']), ('_sourcetype', 'aws:s3:accesslogs'), ('_subsecond', '.892'), ('_time', '2021-02-04T09:53:36.892-08:00'), ('accountID', '[MASKED]'), ('authentication_type', 'AuthHeader'), ('bucket_name', 's3testbucketharish'), ('bucket_owner', 'b5c3c9a2b9e7f8c9c508442a00deb47783d0fd203e0c017f10834561f2607f6a'), ('bytes_sent', '365'), ('cipher_suite', 'ECDHE-RSA-AES128-GCM-SHA256'), ('data_source_name', 'valid_source_testing_sources_1626821762_16937950'), ('date_hour', '17'), ('date_mday', '4'), ('date_minute', '53'), ('date_month', 'february'), ('date_second', '36'), ('date_wday', 'thursday'), ('date_year', '2021'), ('date_zone', '0'), ('error_code', 'AuthorizationHeaderMalformed'), ('etag', '3d0a05c575372ed4770d663fb3f1a04b'), ('host', '457f535a-d25d-47e2-9c1e-5471d8577fb0'), ('host_header', 's3testbucketharish.s3.amazonaws.com'), ('host_id', 'hGyfrBiMikmv9/J1LFIecin4k/B8kBK7i6+FQ0EfNyH3lC8+984oIQja6XOlJtHi9vISwIrfbFU='), ('http_status', '400'), ('index', 'cdce2etestidx'), ('key', 'harishpalin23.txt'), ('lastModified', '1626821763'), ('linecount', '1'), ('object_size', '-'), ('operation', 'REST.DELETE.OBJECT'), ('punct', '__[//:::_+]_..._-__.._._"_/._/."____-__-_"-"_"/.._'), ('referrer', '-'), ('remote_ip', '98.33.33.216'), ('request_id', '647D4303868FF214'), ('request_time', '04/Feb/2021:17:53:36 +0000'), ('request_uri', 'DELETE /harishpalin23.txt HTTP/1.1'), ('requester', '-'), ('signature_version', 'SigV4'), ('source', 's3://premerge-16937950/s3_al_1/1626821762_16937950_premerge_s3_access.log'), ('sourcetype', 'aws:s3:accesslogs'), ('splunk_server', 'idx-i-0972f413e683fd13a.valuable-vulture-6c0.stg.splunkcloud.com'), ('timeendpos', '111'), ('timestartpos', '85'), ('tls_version', 'TLSv1.2'), ('total_time', '3'), ('turn_around_time', '-'), ('user_agent', 'Boto3/1.9.91 Python/3.8.5 Linux/5.4.0-65-generic Botocore/1.12.253'), ('version_id', '-')])
actual Selected {'bucket_owner': 'b5c3c9a2b9e7f8c9c508442a00deb47783d0fd203e0c017f10834561f2607f6a', 'bucket_name': 's3testbucketharish', 'request_time': '04/Feb/2021:17:53:36 +0000', 'remote_ip': '98.33.33.216', 'object_size': '-', 'request_id': '647D4303868FF214', 'operation': 'REST.DELETE.OBJECT', 'key': 'harishpalin23.txt', 'request_uri': 'DELETE /harishpalin23.txt HTTP/1.1', 'http_status': '400', 'error_code': 'AuthorizationHeaderMalformed', 'bytes_sent': '365', 'referrer': '-', 'total_time': '3', 'requester': '-', 'turn_around_time': '-', 'user_agent': 'Boto3/1.9.91 Python/3.8.5 Linux/5.4.0-65-generic Botocore/1.12.253', 'version_id': '-', 'host_id': 'hGyfrBiMikmv9/J1LFIecin4k/B8kBK7i6+FQ0EfNyH3lC8+984oIQja6XOlJtHi9vISwIrfbFU=', 'signature_version': 'SigV4', 'cipher_suite': 'ECDHE-RSA-AES128-GCM-SHA256', 'authentication_type': 'AuthHeader', 'host_header': 's3testbucketharish.s3.amazonaws.com', 'tls_version': 'TLSv1.2', 'source': 's3://premerge-16937950/s3_al_1/1626821762_16937950_premerge_s3_access.log', 'sourcetype': 'aws:s3:accesslogs'}
  """  
import csv
import collections

def csvReader():
    filename = 'data/custom_csv.csv'
    lst = []
    with open(filename,'r') as data:
        for line in csv.DictReader(data):
            lst.append(line)
    d = {}
    d = lst[0]
    print(lst,d)

def jsonReader():
    # Opening JSON file
    resultArr = []
    with open('data/cloudtrail.log.gz') as json_file:
        data = json.load(json_file)
    #print(type(data['Records']))
    for i in range(len(data['Records'])):
        #tempDict = flatten(data['Records'][i])
        tempDict = parse_dict(data['Records'][i])
        resultArr.append(tempDict)
    return resultArr

def flatten(d, parent_key='', sep='.'):
    items = []
    for k, v in d.items():
        new_key = parent_key + sep + k if parent_key else k
        #print(type(new_key),new_key,type(v),v)
        if isinstance(v, dict):
            items.extend(flatten(v, new_key, sep=sep).items())
        elif isinstance(v, list):
            #print("list",v,new_key)
            for i in range(len(v)):
                items.extend(flatten(v[i],new_key+'{}',sep=sep).items())
        else:
            #print(type(new_key),new_key,v)
            for i in range(len(items)):
                print(items[i],new_key)
                if new_key == items[i][0]:
                    print("Items",items[-1][0])
            items.append((new_key, v))
    #print("Items",type(items[-1][0]))
    return dict(items)

def parse_dict(init, lkey='',subnet_val = []):
    ret = {}
    for rkey,val in init.items():
        key = lkey+rkey
        if val is True:
            ret[key] = 'true'
        elif isinstance(val, int):
            ret[key] = str(val)
        elif key == "responseElements" and val == None:
           ret[key] = 'null'
        elif isinstance(val, dict):
            ret.update(parse_dict(val, key+'.',subnet_val))
        elif isinstance(val, list):
            for i in range(len(val)):
                ret.update(parse_dict(val[i],key+'{}'+'.',subnet_val))
        elif key == 'requestParameters.filterSet.items{}.valueSet.items{}.value':
            subnet_val.append(val)
            print("Key FOund")
            ret[key] = subnet_val
        else:
            ret[key] = val   
    return ret
    
def plainText():
    s = ""
    with open("data/custom_plaintxt.txt") as f:
        for line in f:
          s += line
    return s

def cloudFront():
    source_data = ""
    data_dictionary = {}
    data_list = []
    event =  collections.OrderedDict([('_bkt', 'cdce2etestidx~26~36AFB8E5-400C-4443-AD69-E0219D8A33BE'), ('_cd', '26:293680'), ('_indextime', '1626852999'), ('_kv', '1'), ('_raw', '2021-02-05\t03:20:58\tSFO53-C1\t946\t2601:646:9800:2aa0:8a3:4adb:cc58:aa8b\tGET\td116j47k5cstau.cloudfront.net\t/\t307\t-\tMozilla/5.0%20(X11;%20Linux%20x86_64)%20AppleWebKit/537.36%20(KHTML,%20like%20Gecko)%20Chrome/81.0.4044.138%20Safari/537.36\t-\t-\tMiss\tSYfkJdXs5rbhWy5g9AzUPsb7OCKeQ3cIHVquL6iIhBn9LQ8tCw50ag==\td116j47k5cstau.cloudfront.net\thttp\t472\t0.210\t-\t-\t-\tMiss\tHTTP/1.1\t-\t-\t52612\t0.210\tMiss\tapplication/xml\t-\t-\t-'), ('_serial', '0'), ('_si', ['idx-i-0cb3924739f12e95b.valuable-vulture-6c0.stg.splunkcloud.com', 'cdce2etestidx']), ('_sourcetype', 'aws:cloudfront:accesslogs'), ('_subsecond', '.895'), ('_time', '2021-02-04T19:20:58.895-08:00'), ('accountID', '[MASKED]'), ('c_ip', '2601:646:9800:2aa0:8a3:4adb:cc58:aa8b'), ('client_ip', '2601:646:9800:2aa0:8a3:4adb:cc58:aa8b'), ('cs_bytes', '472'), ('cs_cookie', '-'), ('cs_host', 'd116j47k5cstau.cloudfront.net'), ('cs_method', 'GET'), ('cs_protocol', 'http'), ('cs_referer', '-'), ('cs_uri_query', '-'), ('cs_uri_stem', '/'), ('cs_user_agent', 'Mozilla/5.0%20(X11;%20Linux%20x86_64)%20AppleWebKit/537.36%20(KHTML,%20like%20Gecko)%20Chrome/81.0.4044.138%20Safari/537.36'), ('data_source_name', 'valid_source_testing_sources_1626852255_16951547'), ('date', '2021-02-05'), ('date_hour', '3'), ('date_mday', '5'), ('date_minute', '20'), ('date_month', 'february'), ('date_second', '58'), ('date_wday', 'friday'), ('date_year', '2021'), ('date_zone', 'local'), ('edge_location_name', 'San Francisco (California)'), ('etag', '78954d5102cd34f70bfb23387ead44ab'), ('host', 'bb105f3c-ce2b-40db-b47e-3447055dea5a'), ('index', 'cdce2etestidx'), ('lastModified', '1626852256'), ('linecount', '1'), ('punct', '--t::t-tt:::::::tt..t/tt-t/.%(;%%)%/.%(,%%)%/...%/'), ('sc_bytes', '946'), ('sc_status', '307'), ('source', 's3://premerge-16951547/cloudfront_al_1/1626852255_16951547_premerge_cloudfront_access.log.gz'), ('sourcetype', 'aws:cloudfront:accesslogs'), ('splunk_server', 'idx-i-0cb3924739f12e95b.valuable-vulture-6c0.stg.splunkcloud.com'), ('ssl_cipher', '-'), ('ssl_protocol', '-'), ('time', '03:20:58'), ('time_taken', '210'), ('timeendpos', '20'), ('timestartpos', '0'), ('x_edge_location', 'SFO53-C1'), ('x_edge_request_id', 'SYfkJdXs5rbhWy5g9AzUPsb7OCKeQ3cIHVquL6iIhBn9LQ8tCw50ag=='), ('x_edge_response_result_type', 'Miss'), ('x_edge_result_type', 'Miss'), ('x_forwarded_for', '-'), ('x_host_header', 'd116j47k5cstau.cloudfront.net')])
    with open("data/cloudfront_access.log.gz") as f:
        for line in f:
            if "Version" in line:
                continue
            elif "Fields" in line:
                field_Array = line.split()
                del field_Array[0]
                for i in range(len(field_Array)):
                    if "-" in field_Array[i]:
                        field_Array[i] = field_Array[i].replace("-","_")
                    
            
            else:
                temp_Dict = {}
                log_array = line.split()
                for field, value in zip(field_Array, log_array):
                    if field == 'cs_protocol_version' or field == 'fle_status' or field == 'fle_encrypted_fields' or field == 'sc_content_len' or field == 'sc_range_start' or field == 'sc_range_end':  # Need to check on this Field
                        continue
                    elif field == 'cs(Host)':
                        field = 'cs_host'
                    elif field == 'cs(Referer)':
                        field = 'cs_referer'
                    elif field == 'cs(User_Agent)':
                        field = 'cs_user_agent'
                    elif field == 'cs(Cookie)':
                        field = 'cs_cookie'
                    temp_Dict[field] = value
                data_list.append(temp_Dict)
        print("Field Array",field_Array)
        print("Data List", data_list)
    for i in range(len(data_list)):
        print("x-edge-request-id",event['x_edge_request_id'])
        #print(f"Events that needs to be matched{event} and the data coming in {data_list[i]}")
        #print("Event",event)
        if "x-edge-request-id" not in event:
            #print(f"Events that doesnt have x_edge_request_id in the input data {data_list[i]}")
            data_dictionary = {
                "linecount": "2",
                "source": source_data,
                # "s3://premerge-14816426/cloudfront_al_1/1621362009_14816426_premerge_cloudfront_access.log.gz",
                "sourcetype": "aws:cloudfront:accesslogs",
            }
        elif "x-edge-request-id" in event and event["x_edge_request_id"] == data_list[i]["x_edge_request_id"]:
            data_dictionary = data_list[i]
            data_dictionary["source"] = source_data
            data_dictionary["sourcetype"] = "aws:cloudfront:accesslogs"
            data_dictionary["index"] = os.environ.get('SPLUNK_INDEX')
        else:
            print("******* x-edge-request-id NOT found or the value of x-edge-request-id does not match *****")
    return data_dictionary
                

def elasticBlancer():
    field_Array = ["type","timestamp","elb","target","request_processing_time","target_processing_time","response_processing_time","elb_status_code",
                   "target_status_code","received_bytes","sent_bytes","request","user_agent", "target_group_arn",
                   "trace_id","chosen_cert_arn","domain_name","matched_rule_priority","request_creation_time",
                   "actions_executed","error_reason","redirect_url","target","ssl_cipher","ssl_protocol","client_ip","client_port"]
    
    with open("data/11_elasticloadbalancing_appelb.log.gz") as f:
        for line in f:
            fieldDict = {}
            logArray = line.split(" ")
            logArray[12] = logArray[12] + " "+logArray[13]+" "+logArray[14]
            del logArray[13:15]
            del logArray[-3]
            logArray[13] = logArray[13] + " " + logArray[14]+ " " +logArray[15]+ " "+ logArray[16]+ " " +logArray[17]+ " "+logArray[18]
            del logArray[14:19]
            logArray[13] = logArray[13] + " " +logArray[14]+ " "+logArray[15]+ " "+ logArray[16]+ " " +logArray[17]
            del logArray[14:20]
            tempArray = logArray[3]
            del logArray[3]
            print(len(logArray))
            tempArray = tempArray.split(":")
            logArray.append(tempArray[0])
            logArray.append(tempArray[1])
            #print(logArray)
            for field,event in zip(field_Array,logArray):
                if '"' or '\n'in event:
                    event = event.replace('"',"")
                    event = event.replace('\n',"")
                fieldDict[field] = event
            print(fieldDict)

def elasticBlanceraccesslogs():
    field_Array = ["timestamp","elb","backend","request_processing_time","backend_processing_time",
                   "response_processing_time","backend_status_code","elb_status_code","received_bytes","sent_bytes","request",
                   "user_agent","ssl_cipher","ssl_protocol","client_ip","client_port"]
    #print(event)
    with open("data/11_elasticloadbalancing_classicelb.log") as f:
        for line in f:
            fieldDict = {}
            logArray = line.split(" ")
            tempArray = logArray[2]
            del logArray[2]
            logArray[10] = logArray[10] + " "+logArray[11]+" "+logArray[12]
            del logArray[11:13]
            tempArray = tempArray.split(":")
            logArray.append(tempArray[0])
            logArray.append(tempArray[1])
            #logArray[10] = logArray[10].replace('"',"")
            #logArray[11] = logArray[11].replace('"',"")
            print(logArray)
            for field,value in zip(field_Array,logArray):
                if '"' in value:
                    value = value.replace('"',"")
                elif '\n' in value:
                    value = value.replace("\n","")
                fieldDict[field] = value              
            print(fieldDict)
            

def readText():
    with open('data/custom_plaintxt.txt') as f:
        inputtext = f.read()
    print(inputtext)
            
            

class Intervals:
    def __init__(self):
        self.intervals = []
   
    def addInterval(self,interval):
        res = []
        if len(self.intervals) == 0:
            self.intervals.append(interval)
        else:
       
            for idx,intersect in enumerate(self.intervals):
                if intersect[1] < interval[0]:
                    res.append(intersect)
                elif interval[1] < intersect[0]:
                    res.append(interval)
                    self.intervals = res + self.intervals[idx:]
                    return self.intervals
                else:
                    interval[0] = min(intersect[0],interval[0])
                    interval[1] = max(intersect[1], interval[1])
            res.append(interval)
            self.intervals = res  
        print(self.intervals)
        return self.intervals
               
       
    def getTotalCoveredLength(self):
        counter = 0
        for i in range(len(self.intervals)):
            counter += (self.intervals[i][1] - self.intervals[i][0])
       
        return counter
    
c = Intervals()
c.addInterval([3,6])
c.addInterval([8,9])
c.addInterval([1,5])
c.addInterval([1,10])
print(c.getTotalCoveredLength())

#entID': 'fdcc337b-c38b-41f0-bb81-bf469b3f8828', 'awsRegion': 'us-east-2', 'eventCategory': 'Management', 'eventVersion': '1.08', 'responseElements': 'null', 'sourceIPAddress': '98.33.33.216', 'eventSource': 's3.amazonaws.com', 'errorMessage': 'The bucket policy does not exist', 'requestParameters.bucketName': 'cloudtraillogsharish', 'requestParameters.Host': 's3.us-east-2.amazonaws.com', 'requestParameters.policy': '', 'errorCode': 'NoSuchBucketPolicy', 'resources{}.accountId': '[MASKED]', 'resources{}.type': 'AWS::S3::Bucket', 'resources{}.ARN': 'arn:aws:s3:::cloudtraillogsharish', 'userAgent': '[AWSCloudTrail, aws-internal/3 aws-sdk-java/1.11.932 Linux/4.9.230-0.1.ac.223.84.332.metal1.x86_64 OpenJDK_64-Bit_Server_VM/25.275-b01 java/1.8.0_275 vendor/Oracle_Corporation]', 'readOnly': 'true', 'userIdentity.accessKeyId': 'ASIAQSPKYLLEQZQ5G5VX', 'userIdentity.sessionContext.attributes.mfaAuthenticated': 'false', 'userIdentity.sessionContext.attributes.creationDate': '2021-02-04T07:20:54Z', 'userIdentity.accountId': '[MASKED]', 'userIdentity.principalId': 'AIDAQSPKYLLES7RBP4BKD', 'userIdentity.type': 'IAMUser', 'userIdentity.arn': 'arn:aws:iam::[MASKED]:user/hkayarohanam', 'userIdentity.userName': 'hkayarohanam', 'eventType': 'AwsApiCall', 'additionalEventData.SignatureVersion': 'SigV4', 'additionalEventData.CipherSuite': 'ECDHE-RSA-AES128-GCM-SHA256', 'additionalEventData.bytesTransferredIn': '0', 'additionalEventData.AuthenticationMethod': 'AuthHeader', 'additionalEventData.x-amz-id-2': 'E/elmKd9Jj0nDOPiISiwpYXVtmnktsqkKwRbsF7P4j4stTJPpWDwLPCvrhoYugQmuAl4qcHwIqY=', 'additionalEventData.bytesTransferredOut': '313', 'vpcEndpointId': 'vpce-16a4477f', 'requestID': '3144153DD47E294C', 'eventTime': '2021-02-04T11:13:05Z', 'eventName': 'GetBucketPolicy', 'recipientAccountId': '[MASKED]', 'managementEvent': 'true', 'sourcetype': 'aws:cloudtrail', 'source': 's3://premerge-16928367/cloudtrail_1/AWSLogs/[MASKED]/CloudTrail/us-east-2/2021/04/28/1626812999_16928367_premerge_cloudtrail.log.gz', 'index': 'cdce2etestidx'}

"""
Expected Values: {'additionalEventData.AuthenticationMethod': 'AuthHeader', 'additionalEventData.CipherSuite': 'ECDHE-RSA-AES128-GCM-SHA256', 'additionalEventData.SignatureVersion': 'SigV4', 'additionalEventData.bytesTransferredIn': '0', 'additionalEventData.bytesTransferredOut': '313', 'additionalEventData.x-amz-id-2': 'E/elmKd9Jj0nDOPiISiwpYXVtmnktsqkKwRbsF7P4j4stTJPpWDwLPCvrhoYugQmuAl4qcHwIqY=', 'app': 's3.amazonaws.com', 'awsRegion': 'us-east-2', 'command': 'GetBucketPolicy', 'dest': 'cloudtraillogsharish', 'dvc': 's3.amazonaws.com', 'errorCode': 'NoSuchBucketPolicy', 'errorMessage': 'The bucket policy does not exist', 'etag': 'bc97fc7f9fc6f272c740cd3375231385', 'eventCategory': 'Management', 'eventID': 'fdcc337b-c38b-41f0-bb81-bf469b3f8828', 'eventName': 'GetBucketPolicy', 'eventSource': 's3.amazonaws.com', 'eventTime': '2021-02-04T11:13:05Z', 'eventType': 'AwsApiCall', 'eventVersion': '1.08', 'eventtype': 'aws_cloudtrail_errors', 'host': 'a24eff30-52a1-4a24-818a-800acfad2a9d', 'index': 'cdce2etestidx', 'lastModified': '1621362014', 'linecount': '1', 'managementEvent': 'true', 'msg': 'NoSuchBucketPolicy', 'object': 'cloudtraillogsharish', 'object_category': 'unknown', 'object_id': 'cloudtraillogsharish', 'product': 'CloudTrail', 'readOnly': 'true', 'reason': 'The bucket policy does not exist', 'region': 'us-east-2', 'requestID': '3144153DD47E294C', 'requestParameters.Host': 's3.us-east-2.amazonaws.com', 'requestParameters.bucketName': 'cloudtraillogsharish', 'requestParameters.policy': '', 'resources{}.ARN': 'arn:aws:s3:::cloudtraillogsharish', 'resources{}.type': 'AWS::S3::Bucket', 'responseElements': 'null', 'result': 'The bucket policy does not exist', 'result_id': 'NoSuchBucketPolicy', 'signature': 'GetBucketPolicy', 'source': 'cloudtrail_1/AWSLogs/[MASKED]/CloudTrail/us-east-2/2021/04/28/1626214853_16690587_premerge_cloudtrail.log.gz', 'sourceIPAddress': '98.33.33.216', 'sourcetype': 'aws:cloudtrail', 'src': '98.33.33.216', 'src_ip': '98.33.33.216', 'start_time': '2021-02-04T11:13:05Z', 'tag': ['change', 'cloud', 'error'], 'tag::eventtype': ['change', 'cloud', 'error'], 'user': 'AIDAQSPKYLLES7RBP4BKD', 'userAgent': '[AWSCloudTrail, aws-internal/3 aws-sdk-java/1.11.932 Linux/4.9.230-0.1.ac.223.84.332.metal1.x86_64 OpenJDK_64-Bit_Server_VM/25.275-b01 java/1.8.0_275 vendor/Oracle_Corporation]', 'userIdentity.accessKeyId': 'ASIAQSPKYLLEQZQ5G5VX', 'userIdentity.arn': 'arn:aws:iam::[MASKED]:user/hkayarohanam', 'userIdentity.principalId': 'AIDAQSPKYLLES7RBP4BKD', 'userIdentity.sessionContext.attributes.creationDate': '2021-02-04T07:20:54Z', 'userIdentity.sessionContext.attributes.mfaAuthenticated': 'false', 'userIdentity.type': 'IAMUser', 'userIdentity.userName': 'hkayarohanam', 'userName': 'hkayarohanam', 'user_access_key': 'ASIAQSPKYLLEQZQ5G5VX', 'user_agent': '[AWSCloudTrail, aws-internal/3 aws-sdk-java/1.11.932 Linux/4.9.230-0.1.ac.223.84.332.metal1.x86_64 OpenJDK_64-Bit_Server_VM/25.275-b01 java/1.8.0_275 vendor/Oracle_Corporation]', 'user_id': 'AIDAQSPKYLLES7RBP4BKD', 'user_type': 'IAMUser', 'vendor': 'Amazon Web Services', 'vendor_region': 'us-east-2', 'vpcEndpointId': 'vpce-16a4477f'}

OrderedDict([('_bkt', 'cdce2etestidx~26~36AFB8E5-400C-4443-AD69-E0219D8A33BE'), ('_cd', '26:5058'), ('_eventtype_color', 'none'), ('_indextime', '1626214851'), ('_raw', '{"eventID":"fdcc337b-c38b-41f0-bb81-bf469b3f8828","awsRegion":"us-east-2","eventCategory":"Management","eventVersion":"1.08","responseElements":null,"sourceIPAddress":"98.33.33.216","eventSource":"s3.amazonaws.com","errorMessage":"The bucket policy does not exist","requestParameters":{"bucketName":"cloudtraillogsharish","Host":"s3.us-east-2.amazonaws.com","policy":""},"errorCode":"NoSuchBucketPolicy","resources":[{"accountId":"[MASKED]","type":"AWS::S3::Bucket","ARN":"arn:aws:s3:::cloudtraillogsharish"}],"userAgent":"[AWSCloudTrail, aws-internal/3 aws-sdk-java/1.11.932 Linux/4.9.230-0.1.ac.223.84.332.metal1.x86_64 OpenJDK_64-Bit_Server_VM/25.275-b01 java/1.8.0_275 vendor/Oracle_Corporation]","readOnly":true,"userIdentity":{"accessKeyId":"ASIAQSPKYLLEQZQ5G5VX","sessionContext":{"attributes":{"mfaAuthenticated":"false","creationDate":"2021-02-04T07:20:54Z"}},"accountId":"[MASKED]","principalId":"AIDAQSPKYLLES7RBP4BKD",
                                                                                                                                                                 
                                                                                                                                                                 :"IAMUser","arn":"arn:aws:iam::[MASKED]:user/hkayarohanam","userName":"hkayarohanam"},"eventType":"AwsApiCall","additionalEventData":{"SignatureVersion":"SigV4","CipherSuite":"ECDHE-RSA-AES128-GCM-SHA256","bytesTransferredIn":0,"AuthenticationMethod":"AuthHeader","x-amz-id-2":"E/elmKd9Jj0nDOPiISiwpYXVtmnktsqkKwRbsF7P4j4stTJPpWDwLPCvrhoYugQmuAl4qcHwIqY=","bytesTransferredOut":313},"vpcEndpointId":"vpce-16a4477f","requestID":"3144153DD47E294C","eventTime":"2021-02-04T11:13:05Z","eventName":"GetBucketPolicy","recipientAccountId":"[MASKED]","managementEvent":true}'), ('_serial', '0'), ('_si', ['idx-i-0cb3924739f12e95b.valuable-vulture-6c0.stg.splunkcloud.com', 'cdce2etestidx']), ('_sourcetype', 'aws:cloudtrail'), ('_subsecond', '.409'), ('_time', '2021-02-04T03:13:05.409-08:00'), ('accountID', '[MASKED]'), ('additionalEventData.AuthenticationMethod', 'AuthHeader'), ('additionalEventData.CipherSuite', 'ECDHE-RSA-AES128-GCM-SHA256'), ('additionalEventData.SignatureVersion', 'SigV4'), ('additionalEventData.bytesTransferredIn', '0'), ('additionalEventData.bytesTransferredOut', '313'), ('additionalEventData.x-amz-id-2', 'E/elmKd9Jj0nDOPiISiwpYXVtmnktsqkKwRbsF7P4j4stTJPpWDwLPCvrhoYugQmuAl4qcHwIqY='), ('app', 's3.amazonaws.com'), ('awsRegion', 'us-east-2'), ('aws_account_id', '[MASKED]'), ('command', 'GetBucketPolicy'), ('data_source_name', 'valid_source_testing_sources_1626214404_16690587'), ('date_hour', '11'), ('date_mday', '4'), ('date_minute', '13'), ('date_month', 'february'), ('date_second', '5'), ('date_wday', 'thursday'), ('date_year', '2021'), ('date_zone', '0'), ('dest', 'cloudtraillogsharish'), ('dvc', 's3.amazonaws.com'), ('errorCode', 'NoSuchBucketPolicy'), ('errorMessage', 'The bucket policy does not exist'), ('etag', 'bc97fc7f9fc6f272c740cd3375231385'), ('eventCategory', 'Management'), ('eventID', 'fdcc337b-c38b-41f0-bb81-bf469b3f8828'), ('eventName', 'GetBucketPolicy'), ('eventSource', 's3.amazonaws.com'), ('eventTime', '2021-02-04T11:13:05Z'), ('eventType', 'AwsApiCall'), ('eventVersion', '1.08'), ('eventtype', 'aws_cloudtrail_errors'), ('host', '0f025fec-4fc1-4ab2-8e8c-f8772ebedf98'), ('index', 'cdce2etestidx'), ('lastModified', '1626214405'), ('linecount', '1'), ('managementEvent', 'true'), ('msg', 'NoSuchBucketPolicy'), ('object', 'cloudtraillogsharish'), ('object_category', 'unknown'), ('object_id', 'cloudtraillogsharish'), ('product', 'CloudTrail'), ('punct', '{"":"----","":"--","":"","":".","":,"":"...","":".'), ('readOnly', 'true'), ('reason', 'The bucket policy does not exist'), ('recipientAccountId', '[MASKED]'), ('region', 'us-east-2'), ('requestID', '3144153DD47E294C'), ('requestParameters.Host', 's3.us-east-2.amazonaws.com'), ('requestParameters.bucketName', 'cloudtraillogsharish'), ('requestParameters.policy', ''), ('resources{}.ARN', 'arn:aws:s3:::cloudtraillogsharish'), ('resources{}.accountId', '[MASKED]'), ('resources{}.type', 'AWS::S3::Bucket'), ('responseElements', 'null'), ('result', 'The bucket policy does not exist'), ('result_id', 'NoSuchBucketPolicy'), ('signature', 'GetBucketPolicy'), ('source', 's3://premerge-16690587/cloudtrail_1/AWSLogs/[MASKED]/CloudTrail/us-east-2/2021/04/28/1626214404_16690587_premerge_cloudtrail.log.gz'), ('sourceIPAddress', '98.33.33.216'), ('sourcetype', 'aws:cloudtrail'), ('splunk_server', 'idx-i-0cb3924739f12e95b.valuable-vulture-6c0.stg.splunkcloud.com'), ('src', '98.33.33.216'), ('src_ip', '98.33.33.216'), ('start_time', '2021-02-04T11:13:05Z'), ('tag', ['change', 'cloud', 'error']), ('tag::eventtype', ['change', 'cloud', 'error']), ('timeendpos', '1427'), ('timestartpos', '1407'), ('user', 'AIDAQSPKYLLES7RBP4BKD'), ('userAgent', '[AWSCloudTrail, aws-internal/3 aws-sdk-java/1.11.932 Linux/4.9.230-0.1.ac.223.84.332.metal1.x86_64 OpenJDK_64-Bit_Server_VM/25.275-b01 java/1.8.0_275 vendor/Oracle_Corporation]'), ('userIdentity.accessKeyId', 'ASIAQSPKYLLEQZQ5G5VX'), ('userIdentity.accountId', '[MASKED]'), ('userIdentity.arn', 'arn:aws:iam::[MASKED]:user/hkayarohanam'), ('userIdentity.principalId', 'AIDAQSPKYLLES7RBP4BKD'), ('userIdentity.sessionContext.attributes.creationDate', '2021-02-04T07:20:54Z'), ('userIdentity.sessionContext.attributes.mfaAuthenticated', 'false'), ('userIdentity.type', 'IAMUser'), ('userIdentity.userName', 'hkayarohanam'), ('userName', 'hkayarohanam'), ('user_access_key', 'ASIAQSPKYLLEQZQ5G5VX'), ('user_agent', '[AWSCloudTrail, aws-internal/3 aws-sdk-java/1.11.932 Linux/4.9.230-0.1.ac.223.84.332.metal1.x86_64 OpenJDK_64-Bit_Server_VM/25.275-b01 java/1.8.0_275 vendor/Oracle_Corporation]'), ('user_arn', 'arn:aws:iam::[MASKED]:user/hkayarohanam'), ('user_group_id', '[MASKED]'), ('user_id', 'AIDAQSPKYLLES7RBP4BKD'), ('user_name', 'hkayarohanam'), ('user_type', 'IAMUser'), ('vendor', 'Amazon Web Services'), ('vendor_account', '[MASKED]'), ('vendor_region', 'us-east-2'), ('vpcEndpointId', 'vpce-16a4477f')])


'source': 's3://premerge-16697424/cloudtrail_1/AWSLogs/[MASKED]/CloudTrail/us-east-2/2021/04/28/1626220445_16697424_premerge_cloudtrail.log.gz'

'source': 'cloudtrail_1/AWSLogs/[MASKED]/CloudTrail/us-east-2/2021/04/28/1626214853_16690587_premerge_cloudtrail.log.gz'



'source': 's3://premerge-16703372/cloudtrail_1/AWSLogs/[MASKED]/CloudTrail/us-east-2/2021/04/28/1626234138_16703372_premerge_cloudtrail.log.gz'

'source', 's3://premerge-16703372/cloudtrail_1/AWSLogs/[MASKED]/CloudTrail/us-east-2/2021/04/28/1626234138_16703372_premerge_cloudtrail.log.gz'),
    
'source': 's3://premerge-16703372/cloudtrail_1/AWSLogs/[MASKED]/CloudTrail/us-east-2/2021/04/28/1626234512_16703372_premerge_cloudtrail.log.gz'




{'additionalEventData.AuthenticationMethod': 'AuthHeader', 'additionalEventData.CipherSuite': 'ECDHE-RSA-AES128-GCM-SHA256', 'additionalEventData.SignatureVersion': 'SigV4', 'additionalEventData.bytesTransferredIn': '0', 'additionalEventData.bytesTransferredOut': '313', 'additionalEventData.x-amz-id-2': 'E/elmKd9Jj0nDOPiISiwpYXVtmnktsqkKwRbsF7P4j4stTJPpWDwLPCvrhoYugQmuAl4qcHwIqY=', 'app': 's3.amazonaws.com', 'awsRegion': 'us-east-2', 'command': 'GetBucketPolicy', 'dest': 'cloudtraillogsharish', 'dvc': 's3.amazonaws.com', 'errorCode': 'NoSuchBucketPolicy', 'errorMessage': 'The bucket policy does not exist', 'eventCategory': 'Management', 'eventID': 'fdcc337b-c38b-41f0-bb81-bf469b3f8828', 'eventName': 'GetBucketPolicy', 'eventSource': 's3.amazonaws.com', 'eventTime': '2021-02-04T11:13:05Z', 'eventType': 'AwsApiCall', 'eventVersion': '1.08', 'eventtype': 'aws_cloudtrail_errors', 'index': 'cdce2etestidx', 'linecount': '1', 'managementEvent': 'true', 'msg': 'NoSuchBucketPolicy', 'object': 'cloudtraillogsharish', 'object_category': 'unknown', 'object_id': 'cloudtraillogsharish', 'product': 'CloudTrail', 'readOnly': 'true', 'reason': 'The bucket policy does not exist', 'region': 'us-east-2', 'requestID': '3144153DD47E294C', 'requestParameters.Host': 's3.us-east-2.amazonaws.com', 'requestParameters.bucketName': 'cloudtraillogsharish', 'requestParameters.policy': '', 'resources{}.ARN': 'arn:aws:s3:::cloudtraillogsharish', 'resources{}.type': 'AWS::S3::Bucket', 'responseElements': 'null', 'result': 'The bucket policy does not exist', 'result_id': 'NoSuchBucketPolicy', 'signature': 'GetBucketPolicy', 'sourceIPAddress': '98.33.33.216', 'sourcetype': 'aws:cloudtrail', 'src': '98.33.33.216', 'src_ip': '98.33.33.216', 'start_time': '2021-02-04T11:13:05Z', 'tag': ['change', 'cloud', 'error'], 'tag::eventtype': ['change', 'cloud', 'error'], 'user': 'AIDAQSPKYLLES7RBP4BKD', 'userAgent': '[AWSCloudTrail, aws-internal/3 aws-sdk-java/1.11.932 Linux/4.9.230-0.1.ac.223.84.332.metal1.x86_64 OpenJDK_64-Bit_Server_VM/25.275-b01 java/1.8.0_275 vendor/Oracle_Corporation]', 'userIdentity.accessKeyId': 'ASIAQSPKYLLEQZQ5G5VX', 'userIdentity.arn': 'arn:aws:iam::[MASKED]:user/hkayarohanam', 'userIdentity.principalId': 'AIDAQSPKYLLES7RBP4BKD', 'userIdentity.sessionContext.attributes.creationDate': '2021-02-04T07:20:54Z', 'userIdentity.sessionContext.attributes.mfaAuthenticated': 'false', 'userIdentity.type': 'IAMUser', 'userIdentity.userName': 'hkayarohanam', 'userName': 'hkayarohanam', 'user_access_key': 'ASIAQSPKYLLEQZQ5G5VX', 'user_agent': '[AWSCloudTrail, aws-internal/3 aws-sdk-java/1.11.932 Linux/4.9.230-0.1.ac.223.84.332.metal1.x86_64 OpenJDK_64-Bit_Server_VM/25.275-b01 java/1.8.0_275 vendor/Oracle_Corporation]', 'user_id': 'AIDAQSPKYLLES7RBP4BKD', 'user_type': 'IAMUser', 'vendor': 'Amazon Web Services', 'vendor_region': 'us-east-2', 'vpcEndpointId': 'vpce-16a4477f'}
OrderedDict([('_bkt', 'cdce2etestidx~27~BDFB4DAF-A61E-4F04-BE63-ADE4C7740285'), ('_cd', '27:8055'), ('_eventtype_color', 'none'), ('_indextime', '1626237217'), ('_raw', '{"eventID":"fdcc337b-c38b-41f0-bb81-bf469b3f8828","awsRegion":"us-east-2","eventCategory":"Management","eventVersion":"1.08","responseElements":null,"sourceIPAddress":"98.33.33.216","eventSource":"s3.amazonaws.com","errorMessage":"The bucket policy does not exist","requestParameters":{"bucketName":"cloudtraillogsharish","Host":"s3.us-east-2.amazonaws.com","policy":""},"errorCode":"NoSuchBucketPolicy","resources":[{"accountId":"[MASKED]","type":"AWS::S3::Bucket","ARN":"arn:aws:s3:::cloudtraillogsharish"}],"userAgent":"[AWSCloudTrail, aws-internal/3 aws-sdk-java/1.11.932 Linux/4.9.230-0.1.ac.223.84.332.metal1.x86_64 OpenJDK_64-Bit_Server_VM/25.275-b01 java/1.8.0_275 vendor/Oracle_Corporation]","readOnly":true,"userIdentity":{"accessKeyId":"ASIAQSPKYLLEQZQ5G5VX","sessionContext":{"attributes":{"mfaAuthenticated":"false","creationDate":"2021-02-04T07:20:54Z"}},"accountId":"[MASKED]","principalId":"AIDAQSPKYLLES7RBP4BKD","type":"IAMUser","arn":"arn:aws:iam::[MASKED]:user/hkayarohanam","userName":"hkayarohanam"},"eventType":"AwsApiCall","additionalEventData":{"SignatureVersion":"SigV4","CipherSuite":"ECDHE-RSA-AES128-GCM-SHA256","bytesTransferredIn":0,"AuthenticationMethod":"AuthHeader","x-amz-id-2":"E/elmKd9Jj0nDOPiISiwpYXVtmnktsqkKwRbsF7P4j4stTJPpWDwLPCvrhoYugQmuAl4qcHwIqY=","bytesTransferredOut":313},"vpcEndpointId":"vpce-16a4477f","requestID":"3144153DD47E294C","eventTime":"2021-02-04T11:13:05Z","eventName":"GetBucketPolicy","recipientAccountId":"[MASKED]","managementEvent":true}'), ('_serial', '0'), ('_si', ['idx-i-07d27e5c1b2abb013.valuable-vulture-6c0.stg.splunkcloud.com', 'cdce2etestidx']), ('_sourcetype', 'aws:cloudtrail'), ('_subsecond', '.583'), ('_time', '2021-02-04T03:13:05.583-08:00'), ('accountID', '[MASKED]'), ('additionalEventData.AuthenticationMethod', 'AuthHeader'), ('additionalEventData.CipherSuite', 'ECDHE-RSA-AES128-GCM-SHA256'), ('additionalEventData.SignatureVersion', 'SigV4'), ('additionalEventData.bytesTransferredIn', '0'), ('additionalEventData.bytesTransferredOut', '313'), ('additionalEventData.x-amz-id-2', 'E/elmKd9Jj0nDOPiISiwpYXVtmnktsqkKwRbsF7P4j4stTJPpWDwLPCvrhoYugQmuAl4qcHwIqY='), ('app', 's3.amazonaws.com'), ('awsRegion', 'us-east-2'), ('aws_account_id', '[MASKED]'), ('command', 'GetBucketPolicy'), ('data_source_name', 'valid_source_testing_sources_1626236956_16704200'), ('date_hour', '11'), ('date_mday', '4'), ('date_minute', '13'), ('date_month', 'february'), ('date_second', '5'), ('date_wday', 'thursday'), ('date_year', '2021'), ('date_zone', '0'), ('dest', 'cloudtraillogsharish'), ('dvc', 's3.amazonaws.com'), ('errorCode', 'NoSuchBucketPolicy'), ('errorMessage', 'The bucket policy does not exist'), ('etag', 'bc97fc7f9fc6f272c740cd3375231385'), ('eventCategory', 'Management'), ('eventID', 'fdcc337b-c38b-41f0-bb81-bf469b3f8828'), ('eventName', 'GetBucketPolicy'), ('eventSource', 's3.amazonaws.com'), ('eventTime', '2021-02-04T11:13:05Z'), ('eventType', 'AwsApiCall'), ('eventVersion', '1.08'), ('eventtype', 'aws_cloudtrail_errors'), ('host', 'cb0c0013-f582-477c-ba8d-f501c675c620'), ('index', 'cdce2etestidx'), ('lastModified', '1626236957'), ('linecount', '1'), ('managementEvent', 'true'), ('msg', 'NoSuchBucketPolicy'), ('object', 'cloudtraillogsharish'), ('object_category', 'unknown'), ('object_id', 'cloudtraillogsharish'), ('product', 'CloudTrail'), ('punct', '{"":"----","":"--","":"","":".","":,"":"...","":".'), ('readOnly', 'true'), ('reason', 'The bucket policy does not exist'), ('recipientAccountId', '[MASKED]'), ('region', 'us-east-2'), ('requestID', '3144153DD47E294C'), ('requestParameters.Host', 's3.us-east-2.amazonaws.com'), ('requestParameters.bucketName', 'cloudtraillogsharish'), ('requestParameters.policy', ''), ('resources{}.ARN', 'arn:aws:s3:::cloudtraillogsharish'), ('resources{}.accountId', '[MASKED]'), ('resources{}.type', 'AWS::S3::Bucket'), ('responseElements', 'null'), ('result', 'The bucket policy does not exist'), ('result_id', 'NoSuchBucketPolicy'), ('signature', 'GetBucketPolicy'), ('source', 's3://premerge-16704200/cloudtrail_1/AWSLogs/[MASKED]/CloudTrail/us-east-2/2021/04/28/1626236956_16704200_premerge_cloudtrail.log.gz'), ('sourceIPAddress', '98.33.33.216'), ('sourcetype', 'aws:cloudtrail'), ('splunk_server', 'idx-i-07d27e5c1b2abb013.valuable-vulture-6c0.stg.splunkcloud.com'), ('src', '98.33.33.216'), ('src_ip', '98.33.33.216'), ('start_time', '2021-02-04T11:13:05Z'), ('tag', ['change', 'cloud', 'error']), ('tag::eventtype', ['change', 'cloud', 'error']), ('timeendpos', '1427'), ('timestartpos', '1407'), ('user', 'AIDAQSPKYLLES7RBP4BKD'), ('userAgent', '[AWSCloudTrail, aws-internal/3 aws-sdk-java/1.11.932 Linux/4.9.230-0.1.ac.223.84.332.metal1.x86_64 OpenJDK_64-Bit_Server_VM/25.275-b01 java/1.8.0_275 vendor/Oracle_Corporation]'), ('userIdentity.accessKeyId', 'ASIAQSPKYLLEQZQ5G5VX'), ('userIdentity.accountId', '[MASKED]'), ('userIdentity.arn', 'arn:aws:iam::[MASKED]:user/hkayarohanam'), ('userIdentity.principalId', 'AIDAQSPKYLLES7RBP4BKD'), ('userIdentity.sessionContext.attributes.creationDate', '2021-02-04T07:20:54Z'), ('userIdentity.sessionContext.attributes.mfaAuthenticated', 'false'), ('userIdentity.type', 'IAMUser'), ('userIdentity.userName', 'hkayarohanam'), ('userName', 'hkayarohanam'), ('user_access_key', 'ASIAQSPKYLLEQZQ5G5VX'), ('user_agent', '[AWSCloudTrail, aws-internal/3 aws-sdk-java/1.11.932 Linux/4.9.230-0.1.ac.223.84.332.metal1.x86_64 OpenJDK_64-Bit_Server_VM/25.275-b01 java/1.8.0_275 vendor/Oracle_Corporation]'), ('user_arn', 'arn:aws:iam::[MASKED]:user/hkayarohanam'), ('user_group_id', '[MASKED]'), ('user_id', 'AIDAQSPKYLLES7RBP4BKD'), ('user_name', 'hkayarohanam'), ('user_type', 'IAMUser'), ('vendor', 'Amazon Web Services'), ('vendor_account', '[MASKED]'), ('vendor_region', 'us-east-2'), ('vpcEndpointId', 'vpce-16a4477f')])

{'additionalEventData.AuthenticationMethod': 'AuthHeader', 'additionalEventData.CipherSuite': 'ECDHE-RSA-AES128-GCM-SHA256', 'additionalEventData.SignatureVersion': 'SigV4', 'additionalEventData.bytesTransferredIn': '0', 'additionalEventData.bytesTransferredOut': '313', 'additionalEventData.x-amz-id-2': 'E/elmKd9Jj0nDOPiISiwpYXVtmnktsqkKwRbsF7P4j4stTJPpWDwLPCvrhoYugQmuAl4qcHwIqY=', 'app': 's3.amazonaws.com', 'awsRegion': 'us-east-2', 'command': 'GetBucketPolicy', 'dest': 'cloudtraillogsharish', 'dvc': 's3.amazonaws.com', 'errorCode': 'NoSuchBucketPolicy', 'errorMessage': 'The bucket policy does not exist', 'eventCategory': 'Management', 'eventID': 'fdcc337b-c38b-41f0-bb81-bf469b3f8828', 'eventName': 'GetBucketPolicy', 'eventSource': 's3.amazonaws.com', 'eventTime': '2021-02-04T11:13:05Z', 'eventType': 'AwsApiCall', 'eventVersion': '1.08', 'eventtype': 'aws_cloudtrail_errors', 'index': 'cdce2etestidx', 'linecount': '1', 'managementEvent': 'true', 'msg': 'NoSuchBucketPolicy', 'object': 'cloudtraillogsharish', 'object_category': 'unknown', 'object_id': 'cloudtraillogsharish', 'product': 'CloudTrail', 'readOnly': 'true', 'reason': 'The bucket policy does not exist', 'region': 'us-east-2', 'requestID': '3144153DD47E294C', 'requestParameters.Host': 's3.us-east-2.amazonaws.com', 'requestParameters.bucketName': 'cloudtraillogsharish', 'requestParameters.policy': '', 'resources{}.ARN': 'arn:aws:s3:::cloudtraillogsharish', 'resources{}.type': 'AWS::S3::Bucket', 'responseElements': 'null', 'result': 'The bucket policy does not exist', 'result_id': 'NoSuchBucketPolicy', 'signature': 'GetBucketPolicy', 'sourceIPAddress': '98.33.33.216', 'sourcetype': 'aws:cloudtrail', 'src': '98.33.33.216', 'src_ip': '98.33.33.216', 'start_time': '2021-02-04T11:13:05Z', 'tag': ['change', 'cloud', 'error'], 'tag::eventtype': ['change', 'cloud', 'error'], 'user': 'AIDAQSPKYLLES7RBP4BKD', 'userAgent': '[AWSCloudTrail, aws-internal/3 aws-sdk-java/1.11.932 Linux/4.9.230-0.1.ac.223.84.332.metal1.x86_64 OpenJDK_64-Bit_Server_VM/25.275-b01 java/1.8.0_275 vendor/Oracle_Corporation]', 'userIdentity.accessKeyId': 'ASIAQSPKYLLEQZQ5G5VX', 'userIdentity.arn': 'arn:aws:iam::[MASKED]:user/hkayarohanam', 'userIdentity.principalId': 'AIDAQSPKYLLES7RBP4BKD', 'userIdentity.sessionContext.attributes.creationDate': '2021-02-04T07:20:54Z', 'userIdentity.sessionContext.attributes.mfaAuthenticated': 'false', 'userIdentity.type': 'IAMUser', 'userIdentity.userName': 'hkayarohanam', 'userName': 'hkayarohanam', 'user_access_key': 'ASIAQSPKYLLEQZQ5G5VX', 'user_agent': '[AWSCloudTrail, aws-internal/3 aws-sdk-java/1.11.932 Linux/4.9.230-0.1.ac.223.84.332.metal1.x86_64 OpenJDK_64-Bit_Server_VM/25.275-b01 java/1.8.0_275 vendor/Oracle_Corporation]', 'user_id': 'AIDAQSPKYLLES7RBP4BKD', 'user_type': 'IAMUser', 'vendor': 'Amazon Web Services', 'vendor_region': 'us-east-2', 'vpcEndpointId': 'vpce-16a4477f'}
"""
import os
def testing():
    log = os.path.join('data/','AuditLogs.json')
    with open(log) as f:
        aadLog = f.read()
        parsed_log = json.loads(aadLog)
        print(parsed_log)
    
def source_extraction():
    sourceName = 'Endpoint=sb://splkaadlogsehasfowerimzfwe4.servicebus.windows.net/;SharedAccessKeyName=splk-aad-logs-eventhub-auth;SharedAccessKey=HMbHhCjtIj/VasP1nxkPBdK00wze94aFMu19Y4rLbnE='
    print(len(sourceName))
    sourceName = sourceName.split(";")
    return 'azure:westus:'+ sourceName[0][14:41]+":"+"-".join(sourceName[1].split("=")[1].split("-")[0:4])