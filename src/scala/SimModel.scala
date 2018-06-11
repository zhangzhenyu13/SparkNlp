import org.apache.spark.ml.Pipeline
import org.apache.spark.ml.feature._
import org.apache.spark.sql.{DataFrame, Dataset, SparkSession}
import org.apache.spark.ml.classification.{NaiveBayes, RandomForestClassifier}
import org.apache.spark.ml.tuning.{CrossValidator, ParamGridBuilder}
import org.apache.spark.ml.evaluation.BinaryClassificationEvaluator
class SimModel(spark:SparkSession) extends Serializable {
  def EncodeFeatures(data: DataFrame):DataFrame={
    //doc model
    val word2Vec = new Word2Vec()
      .setInputCol("words")
      .setOutputCol("features")
      .setVectorSize(100)
      .setMinCount(0)
      .setMaxIter(5)
      .setStepSize(0.1)
      .setMaxSentenceLength(100)
      .setWindowSize(10)

    val docmodel = word2Vec.fit(data)
    val result=docmodel.transform(data)
    result.show(2)
    return result
    /*
    import spark.implicits._

    result.map{
      x=>
        val label: Int =x(3).asInstanceOf[Int]
        val features: Array[Double] =x(8).asInstanceOf[Array[Double]]
        LabeledPoint(label,Vectors.dense(features))
    }
    * */

  }

  def buildModel(dataframe:DataFrame):DataFrame={

    val data=EncodeFeatures(dataframe)
    print("finihsed: word2vec encoding \n")

    val Array(train,test)=data.randomSplit(Array(0.8,0.2))

    //indexer
    val featureIndexer=new VectorIndexer().setInputCol("features").setOutputCol("indexedfeatures")
    //classifier model
    val rt=new RandomForestClassifier().setLabelCol("label").setFeaturesCol("indexedfeatures").
      setPredictionCol("prediction").setMaxDepth(12).setImpurity("gini")

    //pipe
    val pipeline=new Pipeline().setStages(Array(featureIndexer,rt))
    val parametergrid=new ParamGridBuilder()
      .addGrid(rt.maxDepth,Array(3,5,8,10,12,15))
      .addGrid(rt.impurity,Array("gini","entropy"))
      .build()
    val evaluator=new BinaryClassificationEvaluator().setLabelCol("label")//.setMetricName("f1")
    val crv=new CrossValidator().setEstimator(pipeline).setEvaluator(evaluator).setEstimatorParamMaps(parametergrid).setNumFolds(3)

    val pipemodel=crv.fit(train)
    val result=pipemodel.transform(test)
    result.show(3)
    val roc: Double =evaluator.evaluate(result)
    printf(s"roc=$roc \n")

    var predict=result.select("prediction")
    predict=predict.withColumn("predict",(predict.col("prediction")>0.5))
    predict=predict.withColumn("label",result.col("label"))

    var true_num=0
    predict.rdd.map{x=>
      if (x(1)==x(2)){
        1
      }
      else{
        0
      }
    }.foreach(x=>true_num=true_num+x)
    val testnum=predict.count()
    val acc=true_num/testnum
    printf(s"acc=$acc")

    return result
  }
}

