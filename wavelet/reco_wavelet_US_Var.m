start_idx = 1

reco_baseline = ncread("./nc/daily_US-Var_baseline_16.nc", "nee") + ncread("./nc/daily_US-Var_baseline_16.nc", "gpp")
reco_baseline = reco_baseline(start_idx:end)

reco_default = ncread("./nc/daily_US-Var_default_36.nc", "nee") + ncread("./nc/daily_US-Var_default_36.nc", "gpp")
reco_default = reco_default(start_idx:end)

reco_paw = ncread("./nc/daily_US-Var_nn_paw_12.nc", "nee") + ncread("./nc/daily_US-Var_nn_paw_12.nc", "gpp")
reco_paw = reco_paw(start_idx:end)

reco_whole = ncread("./nc/daily_US-Var_nn_whole_4.nc", "nee") + ncread("./nc/daily_US-Var_nn_whole_4.nc", "gpp")
reco_whole = reco_whole(start_idx:end)

reco_whole_no_lai = ncread("./nc/daily_US-Var_nn_whole_no_lai_21.nc", "nee") + ncread("./nc/daily_US-Var_nn_whole_no_lai_21.nc", "gpp")
reco_whole_no_lai = reco_whole_no_lai(start_idx:end)

reco_h_acm = ncread("./nc/daily_US-Var_gpp_acm_et_nn_33.nc", "nee") + ncread("./nc/daily_US-Var_gpp_acm_et_nn_33.nc", "gpp")
reco_h_acm = reco_h_acm(start_idx:end)

reco_obs = ncread("./nc/US-Var.nc", "RECO")
reco_obs = reco_obs(start_idx:end)
reco_obs = fillmissing(reco_obs, "linear")

DOY = ncread("./nc/US-Var.nc", "DOY")
DOY = DOY(start_idx:end)

set(gcf,'color','w');
colormap("default")
subplot(2, 2, 1)
[wcoh_baseline,~,period_baseline,coi_baseline]  = wcoherence(reco_baseline, reco_obs, days(1), 'phasedisplaythreshold',0.7);
wcoherence(reco_baseline, reco_obs, days(1), 'phasedisplaythreshold',0.7);
hold on 
series_len = size(coi_baseline, 2)
pa=patch([1:1:series_len fliplr(1:1:series_len)], [log2(days(coi_baseline)) zeros(1, series_len)+ max(ylim)], "k");
pa.FaceVertexAlphaData = 0.2;
pa.FaceVertexAlphaData = 0.2;
pa.EdgeColor = "none";
pa.FaceAlpha = 'flat' ; 
title("baseline", FontSize=14)
year_start = find(DOY==1)
year_start(end+1) = length(DOY)
xticks(find(DOY==1))
xticklabels(2002:1:2010)
plot([year_start(6), year_start(6)], ylim, "w--",linewidth=2)
xlabel("Year", FontSize=12)
ylabel("Period (days)", FontSize=12)
hold off

subplot(2, 2, 2)
[wcoh_paw,~,period_paw,coi_paw]  = wcoherence(reco_paw, reco_obs, days(1), 'phasedisplaythreshold',0.7)
wcoherence(reco_paw, reco_obs, days(1), 'phasedisplaythreshold',0.7)
hold on 
series_len = size(coi_paw, 2)
pa=patch([1:1:series_len fliplr(1:1:series_len)], [log2(days(coi_paw)) zeros(1, series_len)+ max(ylim)], "k");
pa.FaceVertexAlphaData = 0.2;
pa.FaceVertexAlphaData = 0.2;
pa.EdgeColor = "none";
pa.FaceAlpha = 'flat' ; 
title("β-nn", FontSize=14)
year_start = find(DOY==1)
year_start(end+1) = length(DOY) + 1
xticks(year_start)
xticklabels(2002:1:2010)
plot([year_start(6), year_start(6)], ylim, "w--",linewidth=2)
xlabel("Year", FontSize=12)
ylabel("Period (days)", FontSize=12)
hold off

subplot(2, 2, 3)
[wcoh_h_acm,~,period_h_acm,coi_h_acm]  = wcoherence(reco_h_acm, reco_obs, days(1), 'phasedisplaythreshold',0.7);
wcoherence(reco_h_acm, reco_obs, days(1), 'phasedisplaythreshold',0.7);
hold on 
series_len = size(coi_h_acm, 2)
pa=patch([1:1:series_len fliplr(1:1:series_len)], [log2(days(coi_h_acm)) zeros(1, series_len)+ max(ylim)], "k")
pa.FaceVertexAlphaData = 0.2;
pa.FaceVertexAlphaData = 0.2;
pa.EdgeColor = "none";
pa.FaceAlpha = 'flat' ; 
title("GPP(ACM)\_ET(NN)", FontSize=14)
year_start = find(DOY==1)
year_start(end+1) = length(DOY) + 1
xticks(year_start)
xticklabels(2002:1:2010)
plot([year_start(6), year_start(6)], ylim, "w--",linewidth=2)
xlabel("Year", FontSize=12)
ylabel("Period (days)", FontSize=12)
hold off

subplot(2, 2, 4)
h=pcolor(1:1:series_len,log2(days(period_h_acm)),wcoh_h_acm-wcoh_baseline)
map = uint8(zeros(512, 3))
map(1:256, 1) = linspace(0, 255, 256)
map(257:end, 1) = 255
map(1:256, 2) = 255
map(257:end, 2) = fliplr(linspace(0, 255, 256))
map(1:256, 3) = 255
map(257:end, 3) = fliplr(linspace(0, 255, 256))
map=flipud(map)

h.EdgeColor = "none";
ax = gca;
ytick=round(pow2(ax.YTick),3);
ax.YTickLabel=ytick;
ax.XLabel.String="Time";
ax.YLabel.String="Period";
ax.Title.String = "GPP(ACM)\_ET(NN) - baseline"
ax.Colormap = map
hcol = colorbar;
hcol.Label.String = "Coherence Difference";
hold on
plot(1:1:series_len,log2(days(coi_h_acm)),"w--",linewidth=2)
pa=patch([1:1:series_len fliplr(1:1:series_len)], [log2(days(coi_h_acm)) zeros(1, series_len)+ max(ylim)], [0.6, 0.6, 0.6])
pa.EdgeColor="none"
year_start = find(DOY==1)
year_start(end+1) = length(DOY) + 1
xticks(year_start)
xticklabels(2002:1:2010)
plot([year_start(6), year_start(6)], ylim, "w--",linewidth=2)
xlabel("Year", FontSize=12)
ylabel("Period (days)", FontSize=12)
hold off



