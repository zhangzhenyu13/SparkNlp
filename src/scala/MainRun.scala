import org.apache.spark.sql.SparkSession
import org.apache.spark.sql.types.StructType



object MainRun{
  val spark=SparkSession.builder()
    //.master("local[*]")
    .appName("NlpDecision").getOrCreate()
  val pathRoot="hdfs://bd60:9000/"

}
