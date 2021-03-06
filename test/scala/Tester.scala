object Tester  {
  def main(args: Array[String]): Unit = {
    val spark=MainRun.spark

    val dloader=new LoadData(spark)
    val data=dloader.readCSV(List("atec_nlp/atec_nlp_sim_train.csv","atec_nlp/atec_nlp_sim_train_add.csv"))
    val num=data.count()
    val num_t=data.select("words").count()
    print(s"data entry size=$num \n")

    val nlpDecision=new SimModel(spark)
    nlpDecision.buildModel(data)
  }
}
