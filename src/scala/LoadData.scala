import org.apache.spark.sql.{DataFrame, Dataset}
import org.apache.spark.sql.SparkSession
import org.ansj.recognition.impl.StopRecognition
import org.ansj.splitWord.analysis.NlpAnalysis
import org.apache.spark.ml.feature.LabeledPoint


case class DataFormat(id:Long,sentence1:String,sentence2:String,
                      label:Int,
                      words1:Array[String],words2:Array[String],
                      words:Array[String],sentence:String)

class LoadData(val spark: SparkSession) extends Serializable {

  def parseString(str:String):DataFormat={
    val fields=str.split("\t")
    //print("fields size",fields.size)
    assert(fields.size==4)
    val id=fields(0).toLong
    val sentence1=fields(1)
    val sentence2=fields(2)
    val words1=SplitChineseSentence(fields(1))
    val words2=SplitChineseSentence(fields(2))
    val words=words1.union(words2)
    val sentence="%s %s".format(fields(1), fields(2))
    val label=fields(3).toInt

    DataFormat(id,sentence1,sentence2,
      label,
      words1,words2,
      words, sentence)
  }

  def SplitChineseSentence(str:String):Array[String]={

    val filers=new StopRecognition()
    filers.insertStopNatures("w")
    val ss: Array[String] =NlpAnalysis.parse(str).recognition(filers).toStringWithOutNature(" ").split(" ")

    return ss

  }

  def readCSV(paths:List[String]):DataFrame={
    import spark.implicits._
    val pathRoot=MainRun.pathRoot
    var data: Dataset[String] =spark.read.textFile(pathRoot+paths.head)
    var maxRI=paths.length-1
    for(i<- 1 to maxRI){
      val data1=spark.read.textFile(pathRoot+paths(i))
      data=data.union(data1)
    }

    var df=data.map(parseString).toDF()

    return df

  }

}
