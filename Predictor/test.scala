package wtytest
import java.util.Calendar;
import io.gatling.core.Predef._
import io.gatling.jdbc.Predef._
import io.gatling.http.Predef._
import scala.io.Source
import scala.concurrent.duration._
import scala.collection.mutable.ArraySeq
import io.gatling.core.structure.PopulationBuilder

class SimpleTest1 extends Simulation {
    val httpProtocol = http
    .baseUrl("http://139.9.57.167:59080")
    .acceptHeader("text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0
.8")
    .doNotTrackHeader("1")
    .acceptLanguageHeader("en-US,en;q=0.5")
    .acceptEncodingHeader("gzip, deflate")
    .userAgentHeader("Mozilla/5.0 (Macintosh; Intel Mac OS X 10.8; rv:16.0) Geck
o/20100101 Firefox/16.0")
    val source = Source.fromFile("wtytest","UTF-8")
    val lines = source.getLines().toArray
    source.close()
    val scn = scenario("RandomWorkLoad").during(1) {
      exec(http("send request").get("/productpage"))
    }

    def scnList() = {
        var step = 1;
        var scnList = new ArraySeq[PopulationBuilder](2880/step);
        for (i <- 0 until 2880 by step) {
            var rn = 0;
            rn = lines(i/60).toDouble.toInt
            var scen = scenario("Normal Access" + i)
                .exec(http("send request").get("/productpage"))
                .inject(
                    nothingFor(i seconds),
                    atOnceUsers(rn)
                );
            scnList(i/step) = scen;
        }
        scnList;
    }
    var now = Calendar.getInstance()
    var currentMinute = now.get(Calendar.MINUTE)
    while( currentMinute < 14){
        now = Calendar.getInstance()
        currentMinute = now.get(Calendar.MINUTE)
    }
    println("pass")
    setUp(scnList: _*).protocols(httpProtocol)
}
