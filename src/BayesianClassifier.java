import java.io.*;
import java.lang.reflect.Array;
import java.nio.file.Files;
import java.nio.file.NoSuchFileException;
import java.util.*;
import java.util.regex.Matcher;
import java.util.regex.Pattern;


/**
 * Bayesian Classifier for spam email detection.
 *
 * This class implement the train method and isSpam method to train a classifier
 * and predict a given email.
 * <p>
 *     The training process works as follow:
 *     1. Learn the vocabulary of both spam and ham emails by extracting and cleaning terms from email text.
 *     2. Count and store the occurrence and frequency of terms into HashMap,
 *     3. Compute the spamicity P(Spam|Term) of each term according to their counted frequency, and stores it into HashMap.
 *
 *     The predict process works as follow:
 *     1. Transform text into terms and clean them,
 *     2. Fetch the spamicities of terms from HashMap, and sort them by value of spamicities,
 *     3. Extract the number of considered terms according to their absolute spamicity compared to 0.5.
 *     4. Compute the probability of spam message and predict results according to the threshold.
 *
 * <p>
 *     Worth to mention:
 *          1. A list of stop words are used, and proved to be useful in increasing precision and cut vocabulary size,
 *          2. The Porter stemming algorithm is used as the stemmer, and proved to be beneficial to increase precision,
 *          3. Dropping terms with extremely rare occurrence in documents (outliers) is helpful.
 *
 * @author Te Tan
 * @version 1.0
 * @since 2018-05-17
 *
 * The Porter stemming algorithm is used as the stemmer.
 * @see Stemmer
 */
public class BayesianClassifier {

    //**************************************************
    //  Constants
    //**************************************************
    /**
     * Number of terms considered when predicting a tested file,
     * terms are sorted by their absolute value compared to 0.5 */
    private static final int NR_OF_TERMS_CONSIDER = 20;
    /**
     * The probability threshold used to decide a spam email */
    private static final double SPAM_PROBAB_THRESHOLD = 0.7;
    /**
     * The threshold of occurrence under which term will be discarded,
     * If a term occurs only in a few files, we will not consider it. */
    private static final int OCCUR_THRESHOLD_OF_DISCARD_TERM = 5;
    /**
     * Path to load stop words */
    private static final String STOP_WORDS_PATH = "stop_words.txt";

    //**************************************************
    //  Fields
    //**************************************************
    /**
     * The map to store spam term counts and computed term spamicity P(Spam|Term) */
    private HashMap<String, Double> spamTermMap;
    /**
     * The map to store ham term counts */
    private HashMap<String, Double> hamTermMap;
    /**
     * The map to store how many emails contain this term
     */
    private HashMap<String, Integer> termOccursInEmailMap;
    /**
     * A hash set to store stop words */
    private Set<String> stopWords;

    //**************************************************
    //  Constructor
    //**************************************************
    public BayesianClassifier() {
        this.spamTermMap = new HashMap<>();
        this.hamTermMap = new HashMap<>();
        this.termOccursInEmailMap = new HashMap<>();
        this.stopWords = this.loadStopWords();
    }

    //**************************************************
    //  Private methods
    //**************************************************
    /**
     * load stop words from Path defined by STOP_WORDS_PATH
     * @return a HashSet of stop words
     */
    private Set<String> loadStopWords() {
        Set<String> stopWords = new HashSet<>();
        try {
            InputStream input = getClass().getResourceAsStream(STOP_WORDS_PATH);
            BufferedReader buff = new BufferedReader(new InputStreamReader(input));
            String line = null;
            while ((line = buff.readLine()) != null) {
                Collections.addAll(stopWords, line.split(","));
            }
        } catch (Exception e) {
            e.printStackTrace();
        }

        return stopWords;
    }

    /**
     * filter files without right format, which should end with txt
     * @param initialFiles
     * @return a File array with legal names
     */
    private File[] filterFiles(File[] initialFiles) {

        Vector listOfFiles = new Vector();
        for(int i = 0; i<initialFiles.length; i++){
            if (initialFiles[i].getName().endsWith(".txt")) {
                listOfFiles.addElement(initialFiles[i]);
            }
        }

        File[] fileArray = new File[listOfFiles.size()];
        listOfFiles.toArray(fileArray);
        return fileArray;

   }

    /**
     * Get a buffered reader from a single file
     * @param file
     * @return a BufferedReader when file is read successfully, or null if not.
     */
   private BufferedReader getReaderFromFile(File file) {
       try {
           InputStream fin = new FileInputStream(file);
           return new BufferedReader(new InputStreamReader(fin));
       } catch (FileNotFoundException e) {
           e.printStackTrace();
           return null;
       }
   }

    /**
     * Get a buffered reader from an array of files,
     * files will be combined by SequenceInputStream,
     * @param files
     * @return a BufferedReader instance decorated the SequenceInputStream
     */
   private BufferedReader getReaderFromFiles(File[] files) {
        ArrayList<InputStream> fileInputs = new ArrayList<>();
       int numberOfFiles = 0;

        for (File file: files) {
            try {
                InputStream fin = new FileInputStream(file);
                fileInputs.add(fin);
                numberOfFiles++;
            } catch (FileNotFoundException e) {
                e.printStackTrace();
            }
        }

       Enumeration inputEnum = Collections.enumeration(fileInputs);
       System.out.println(numberOfFiles+" files found in spam training folder");
       InputStream seqInputs = new SequenceInputStream(inputEnum);
       return new BufferedReader(new InputStreamReader(seqInputs));
   }

    /**
     * extract and stem alphabeta words from a string
     * The Porter stemming algorithm is used.
     *
     * @param line a string to be transformed and cleaned
     * @return an ArrayList of cleaned strings
     */
   private ArrayList<String> cleanString(String line) {
       ArrayList<String> strings = new ArrayList<>();

       Pattern pattern = Pattern.compile("[a-zA-Z]+");
       Matcher matcher = pattern.matcher(line);
       while (matcher.find()) {
           String tobeStemmed = matcher.group();
           if (this.stopWords.contains(tobeStemmed)) {
               continue;
           }

           Stemmer stemmer = new Stemmer();
           stemmer.add(tobeStemmed.toCharArray(), tobeStemmed.toCharArray().length);
           stemmer.stem();
           strings.add(stemmer.toString());
       }
       return strings;
   }

    /**
     * Compute the term spamicity,
     * we consider P(Spam) and P(Ham) both as 0.5, with no bias.
     */
    private void computeProbab() {
        HashSet<String> spamTerms = new HashSet<>(this.spamTermMap.keySet());
        HashSet<String> hamTerms = new HashSet<>(this.hamTermMap.keySet());

        /** merge the vocabulary of hamTermMap to spamTermMap */
        for (String term : hamTerms) {
            if (!spamTerms.contains(term)) {
                this.spamTermMap.put(term, 0.0);
                spamTerms.add(term);
            }
        }

        /** calculate term value counts */
        double spamSum = 0;
        for (double v : this.spamTermMap.values()) {
            spamSum += v;
        }

        double hamSum = 0;
        for (double v : this.hamTermMap.values()) {
            hamSum += v;
        }

        for (String term : spamTerms) {
            Integer termOccurs = this.termOccursInEmailMap.getOrDefault(term, 0);
            if (termOccurs <= OCCUR_THRESHOLD_OF_DISCARD_TERM) {
                this.spamTermMap.remove(term);
                continue;
            }

            double spamFreq = this.spamTermMap.get(term) / spamSum;
            double hamFreq = this.hamTermMap.getOrDefault(term, 0.0) / hamSum;

            double spamTermProbab = spamFreq/(spamFreq + hamFreq);
            this.spamTermMap.put(term, spamTermProbab);
        }
    }


    //**************************************************
    //  Public methods
    //**************************************************
    /**
     * Learn and store the vocabulary of spam and ham emails,
     * compute and feed term counts and spamicity into spamTermMap and hamTermMap
     *
     * @param spamTrainingFolder the folder holding spam emails for training.
     * @param hamTrainingFolder the folder holding ham emails for training.
     */
   public void train(String spamTrainingFolder, String hamTrainingFolder)  {

        File spamTrainingDirectory = new File(spamTrainingFolder);
                if (!spamTrainingDirectory.exists()){
            System.out.println("ERR: The Spam Training Directory does not exist");
            return;
        }

        File hamTrainingDirectory = new File(hamTrainingFolder);
        if (!hamTrainingDirectory.exists()){
            System.out.println("ERR: The Ham Training Directory does not exist");
            return;
        }


        File spamFiles[] = filterFiles(spamTrainingDirectory.listFiles());

       try {
//            BufferedReader spamReader = getReaderFromFiles(spamFiles);
           /**
            * Learn the vocabulary of spam emails, counting the occurrence of terms
            */
            int fileCounter = 0;
            for (File file : spamFiles) {
                BufferedReader spamReader = getReaderFromFile(file);

                HashSet<String> termOccur = new HashSet<>();
                String spamLine = null;
                while ((spamLine = spamReader.readLine()) != null) {
                    ArrayList<String> strings = cleanString(spamLine);

                    for (String str : strings) {
                        double value = this.spamTermMap.getOrDefault(str, 0.0);
                        this.spamTermMap.put(str, value + 1);
                        termOccur.add(str);
                    }
                }
                spamReader.close();
                fileCounter++;
                for (String str : termOccur) {
                    Integer occurs = this.termOccursInEmailMap.getOrDefault(str, 0);
                    this.termOccursInEmailMap.put(str, occurs + 1);
                }
            }
            System.out.println(fileCounter + "  spam training files read.");

            File hamFiles[] = filterFiles(hamTrainingDirectory.listFiles());
//            BufferedReader hamReader = getReaderFromFiles(hamFiles);

           /**
            * Learn the vocabulary of ham emails, counting the occurrence of terms
            */
            fileCounter = 0;
            for (File file : hamFiles) {
                BufferedReader hamReader = getReaderFromFile(file);

                HashSet<String> termOccur = new HashSet<>();
                String hamLine = null;
                while ((hamLine = hamReader.readLine()) != null) {
                    ArrayList<String> strings = cleanString(hamLine);

                    for (String str : strings) {
                        double value = this.hamTermMap.getOrDefault(str, 0.0);
                        this.hamTermMap.put(str, value + 1);
                        termOccur.add(str);
                    }
                }
                hamReader.close();
                fileCounter++;
                for (String str : termOccur) {
                    Integer occurs = this.termOccursInEmailMap.getOrDefault(str, 0);
                    this.termOccursInEmailMap.put(str, occurs + 1);
                }
            }
            System.out.println(fileCounter + "  ham training files read.");
        } catch (IOException e) {
            e.printStackTrace();
            System.out.println("Error when reading file !");
        }

        this.computeProbab();

    }


    public void test(String spamTestingFolder, String hamTestingFolder) {

        File spamTestingDirectory = new File(spamTestingFolder);
        if (!spamTestingDirectory.exists()){
            System.out.println("ERR: The Spam Testing Directory does not exist");
            return;
        }

        File hamTestingDirectory = new File(hamTestingFolder);
        if (!hamTestingDirectory.exists()){
            System.out.println("ERR: The Ham Testing Directory does not exist");
            return;
        }

        System.out.println("Testing phase:");
        
        int allSpam = 0;
        int SpamClassifiedAsHam = 0; //Spams incorrectly classified as Hams

        File spamFiles[] = filterFiles(spamTestingDirectory.listFiles());
        for (File f : spamFiles) {
            allSpam++;
            if (!isSpam(f))
                SpamClassifiedAsHam++;

        }

        int allHam = 0;
        int HamClassifiedAsSpam = 0; //Hams incorrectly classified as Spams
        
        File hamFiles[] = filterFiles(hamTestingDirectory.listFiles());
        for (File f : hamFiles) {
            allHam++;
            if (isSpam(f))
                HamClassifiedAsSpam++;

        }

        System.out.println("###_DO_NOT_USE_THIS_###Spam = "+allSpam);
        System.out.println("###_DO_NOT_USE_THIS_###Ham = "+allHam);
        System.out.println("###_DO_NOT_USE_THIS_###SpamClassifAsHam = "+SpamClassifiedAsHam);
        System.out.println("###_DO_NOT_USE_THIS_###HamClassifAsSpam = "+HamClassifiedAsSpam);


    }


    /**
     * predict whether an email is spam
     * @param f the file to be tested
     * @return true if the file is predicted as spam, false if not
     */
    public boolean isSpam(File f) {
        try {
            BufferedReader reader = new BufferedReader(new InputStreamReader(new FileInputStream(f)));
            HashSet<String> tockens = new HashSet<>();
            ArrayList<Double> termProbabs = new ArrayList<Double>();

            String line = null;
            while ((line = reader.readLine()) != null) {
                ArrayList<String> strings = cleanString(line);
                tockens.addAll(strings);
            }

            reader.close();

            for (String str : tockens) {
                if (this.spamTermMap.containsKey(str)) {
                    termProbabs.add(this.spamTermMap.get(str));
                }
//                termProbabs.add(this.spamTermMap.getOrDefault(str, UNKNOWN_WORD_DEFAULT_PROBAB));
            }

            PriorityQueue<Double> descendSortedProbabs = new PriorityQueue<Double>(termProbabs.size(), Collections.reverseOrder());
            PriorityQueue<Double> ascendSortedProbabs = new PriorityQueue<Double>(termProbabs);
            descendSortedProbabs.addAll((Collection<? extends Double>) termProbabs.clone());

            double logSum = 0;
            for (int i = 0; i < NR_OF_TERMS_CONSIDER; i++) {
                Double highProbab = descendSortedProbabs.poll();
                Double lowProbab = ascendSortedProbabs.poll();
                if (highProbab != null) {
                    if (highProbab == 1.0) {highProbab = 0.9999;}
                    logSum += Math.log(1 - highProbab) - Math.log(highProbab);
                }
                if (lowProbab != null) {
                    if (lowProbab == 0.0) {lowProbab = 0.0001;}
                    logSum += Math.log(1 - lowProbab) - Math.log(lowProbab);
                }
            }
            double p = 1 / (1 + Math.exp(logSum));

            if (p > SPAM_PROBAB_THRESHOLD) {
                return true;
            } else { return false;}

        } catch (NoSuchFileException e) {
            e.printStackTrace();
            System.out.println("No such element found.");
            return true;
        } catch (FileNotFoundException e) {
            e.printStackTrace();
            System.out.println("File not found :" + f.getName());
            return true;
        } catch (IOException e) {
            e.printStackTrace();
            System.out.println("Error when reading file !");
            return true;
        }
    }
    
}
