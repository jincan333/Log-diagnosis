% train_text.csv 训练文本以server name升序排列
textPath = 'pre_contest\dataset\train_text.csv';
TEXT = readtable(textPath);
% train_label.csv 两个label文件合并以server name升序排列
labelPath = 'pre_contest\dataset\train_label.csv';
LABEL = readtable(labelPath);
LABEL = unique(LABEL);
serverList = TEXT.sn;
msg = cell(height(LABEL),1);
label = zeros(height(LABEL),1);
% LABEL(6109,:) = [];
for iter = 1:height(LABEL)
    disp(iter)
    server_iter = LABEL.sn{iter, 1};
    idx = strcmp(serverList,server_iter);
    if sum(strcmp(LABEL.sn,server_iter)) == 1
        msg{iter,1} = ExtractText2(TEXT.msg(idx,1));
        label(iter) = LABEL.label(iter);
    else
        faultTime = time2num(TEXT.time(idx));
        try
            clusterResult = kmeans(faultTime,sum(strcmp(LABEL.sn,server_iter)));
        catch
            clusterResult = kmeans(faultTime,length(faultTime));
        end
        time_iter = LABEL.fault_time(iter);
        [~, minIdx] = min(abs(time2num(time_iter - TEXT.time(idx))));
        subInd = clusterResult == clusterResult(minIdx);
        msgTmp = TEXT.msg(idx,1);
        msg{iter,1} = ExtractText2(msgTmp(subInd,1));
        label(iter) = LABEL.label(iter);
    end
end

function msg_processed = ExtractText2(msg_raw)
msg_processed = unique(msg_raw);
msg_processed = strjoin(msg_processed);


            