#exports avaraged cosine simularity data used to make the bottom part of figure 3.1 and figure 3.2
data_path <- "C:/Users/Matthijs/Desktop/CSV/"
library(readr)
library(ggplot2)
library(reshape2)

#fill in before running!
ms<-2999  #time in ms-1
probes<-c(90, 93,97,102,108,115,123,132)
runs<-99 #amount of runs-1


#CUED MODULE

#column names
names<-list("t","0","3","7","12","18","25","33","42","ping")
#load all csv files in data frame
dat <- data.frame(t=array(0:ms))
#possible orientation differences
for (probe in probes) {
  for (run in 0:runs){
    name <-paste("81_cs_mem_cued_stim_",probe,"_run_",run,".csv",sep = "")
    newcol<-read_csv(paste0(data_path,name),col_names=FALSE,col_types=cols(X1 = col_double()))
    dat <- cbind(dat,newcol)
  }
}
#impulse
for (run in 0:runs){
  name <-paste("81_cs_mem_cued_stim_",999,"_run_",run,".csv",sep = "")
  newcol<-read_csv(paste0(data_path,name),col_names=FALSE,col_types=cols(X1 = col_double()))
  dat <- cbind(dat,newcol)
}
  
datmean <- data.frame(t<-array(0:ms))

for (probe in 0:8){
  datmean<-cbind(datmean,rowMeans(dat[(probe*(runs+1)+2):(probe*(runs+1)+runs+2)]))
  
}
colnames(datmean) <- names
datmelted <-melt(datmean,  id.vars = 't', variable.name = 'probe')

#check if averaging went well with ggplot
ggplot(datmelted, aes(t,value)) +
  geom_line(aes(colour = probe), size= 0.5) + 
  theme_classic()+
  guides(colour=guide_legend(ncol=2,title='Orientation difference \nbetween memory \nitem and probe')) +
  ggtitle("Representation during probe")+
  ylab('Output decision ensemble')+
  xlab('Time (ms)')

#export averaged data to CSV
write.table(datmean,'C:/Users/Matthijs/Desktop/CSV_out/mem_cued.csv', sep = ",",col.names = FALSE,row.names = FALSE)

#-----------------------------------------
#UNCUED MODULE

#column names
names<-list("t","0","3","7","12","18","25","33","42","ping")
#load all csv files in data frame
dat <- data.frame(t=array(0:ms))
#possible orientation differences
for (probe in probes) {
  for (run in 0:runs){
    name <-paste("81_cs_mem_uncued_stim_",probe,"_run_",run,".csv",sep = "")
    newcol<-read_csv(paste0(data_path,name),col_names=FALSE,col_types=cols(X1 = col_double()))
    dat <- cbind(dat,newcol)
  }
}
#impulse
for (run in 0:runs){
  name <-paste("81_cs_mem_uncued_stim_",999,"_run_",run,".csv",sep = "")
  newcol<-read_csv(paste0(data_path,name),col_names=FALSE,col_types=cols(X1 = col_double()))
  dat <- cbind(dat,newcol)
}

datmean <- data.frame(t<-array(0:ms))

for (probe in 0:8){
  datmean<-cbind(datmean,rowMeans(dat[(probe*(runs+1)+2):(probe*(runs+1)+runs+2)]))
  
}
colnames(datmean) <- names
datmelted <-melt(datmean,  id.vars = 't', variable.name = 'probe')

#check if averaging went well with ggplot
ggplot(datmelted, aes(t,value)) +
  geom_line(aes(colour = probe), size= 0.5) + 
  theme_classic()+
  guides(colour=guide_legend(ncol=2,title='Orientation difference \nbetween memory \nitem and probe')) +
  ggtitle("Representation during probe")+
  ylab('Output decision ensemble')+
  xlab('Time (ms)')

#export averaged data to CSV
write.table(datmean,'C:/Users/Matthijs/Desktop/CSV_out/mem_uncued.csv', sep = ",",col.names = FALSE,row.names = FALSE)

