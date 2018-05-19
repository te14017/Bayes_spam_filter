import java.io.File;

/**
 * Created by IntelliJ IDEA.
 * User: verman
 * Date: 4/23/12
 * Time: 3:02 PM
 * To change this template use File | Settings | File Templates.
 */
public class SpamFilter {

    public static void main(String[] args) {

        if ( (args.length < 4) ){
            System.out.println("Insufficient input arguments. Input format should be:");
            System.out.println("spamTrainingFolder, hamTrainingFolder, spamTestingFolder, hamTestingFolder");
                   
        }
        else {

            BayesianClassifier bc = new BayesianClassifier();

            String spamTrainingFolder = args[0];
            String hamTrainingFolder = args[1];
            
            bc.train(spamTrainingFolder, hamTrainingFolder);

            String spamTestingFolder = args[2];
            String hamTestingFolder = args[3];

            bc.test(spamTestingFolder, hamTestingFolder);




        }


    }



    
}
