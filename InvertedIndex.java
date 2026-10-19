import java.io.BufferedReader;
import java.io.IOException;
import java.io.InputStreamReader;
import java.nio.charset.StandardCharsets;
import java.util.HashSet;
import java.util.Locale;
import java.util.Map;
import java.util.Set;
import java.util.TreeMap;

import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.LongWritable;
import org.apache.hadoop.io.NullWritable;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.mapreduce.Job;
import org.apache.hadoop.mapreduce.Mapper;
import org.apache.hadoop.mapreduce.Reducer;
import org.apache.hadoop.mapreduce.lib.input.FileInputFormat;
import org.apache.hadoop.mapreduce.lib.input.FileSplit;
import org.apache.hadoop.mapreduce.lib.output.FileOutputFormat;

public class InvertedIndex {

    private static final String STOPWORDS_PATH =
            "invertedindex.stopwords.path";

    public static class IndexMapper
            extends Mapper<LongWritable, Text, Text, Text> {

        private final Set<String> stopWords = new HashSet<>();
        private final Text outputKey = new Text();
        private final Text outputValue = new Text();
        private String documentName;

        @Override
        protected void setup(Context context)
                throws IOException {

            FileSplit split = (FileSplit) context.getInputSplit();
            documentName = split.getPath().getName();

            Configuration configuration =
                    context.getConfiguration();

            String stopwordsFile =
                    configuration.get(STOPWORDS_PATH);

            if (stopwordsFile == null ||
                    stopwordsFile.trim().isEmpty()) {
                throw new IOException(
                        "Stopwords file path is not configured");
            }

            Path stopwordsPath = new Path(stopwordsFile);
            FileSystem fileSystem =
                    stopwordsPath.getFileSystem(configuration);

            try (BufferedReader reader = new BufferedReader(
                    new InputStreamReader(
                            fileSystem.open(stopwordsPath),
                            StandardCharsets.UTF_8))) {

                String line;

                while ((line = reader.readLine()) != null) {
                    String[] words = line
                            .toLowerCase(Locale.ROOT)
                            .split("[^a-z]+");

                    for (String word : words) {
                        if (!word.isEmpty()) {
                            stopWords.add(word);
                        }
                    }
                }
            }
        }

        @Override
        protected void map(
                LongWritable key,
                Text value,
                Context context)
                throws IOException, InterruptedException {

            String[] words = value.toString()
                    .toLowerCase(Locale.ROOT)
                    .split("[^a-z]+");

            for (String word : words) {
                if (word.length() < 2 ||
                        stopWords.contains(word)) {
                    continue;
                }

                outputKey.set(word);
                outputValue.set(documentName + "\t1");

                context.write(outputKey, outputValue);
            }
        }
    }

    public static class IndexReducer
            extends Reducer<Text, Text, Text, NullWritable> {

        @Override
        protected void reduce(
                Text key,
                Iterable<Text> values,
                Context context)
                throws IOException, InterruptedException {

            Map<String, Integer> documentCounts =
                    new TreeMap<>();

            int total = 0;

            for (Text value : values) {
                String item = value.toString();
                int position = item.lastIndexOf('\t');

                if (position <= 0 ||
                        position == item.length() - 1) {
                    continue;
                }

                String documentName =
                        item.substring(0, position);

                int count = Integer.parseInt(
                        item.substring(position + 1));

                documentCounts.put(
                        documentName,
                        documentCounts.getOrDefault(
                                documentName, 0) + count);

                total += count;
            }

            StringBuilder result = new StringBuilder();

            for (Map.Entry<String, Integer> entry
                    : documentCounts.entrySet()) {

                result.append("<")
                        .append(entry.getKey())
                        .append(",")
                        .append(entry.getValue())
                        .append(">;");
            }

            result.append("<total,")
                    .append(total)
                    .append(">.");

            context.write(
                    new Text(key.toString() + " " + result),
                    NullWritable.get());
        }
    }

    public static void main(String[] args)
            throws Exception {

        if (args.length != 3) {
            System.err.println(
                    "Usage: InvertedIndex " +
                    "<input> <output> <stopwords>");
            System.exit(2);
        }

        Configuration configuration =
                new Configuration();

        configuration.set(STOPWORDS_PATH, args[2]);

        Job job = Job.getInstance(
                configuration,
                "document inverted index");

        job.setJarByClass(InvertedIndex.class);

        job.setMapperClass(IndexMapper.class);
        job.setReducerClass(IndexReducer.class);

        job.setMapOutputKeyClass(Text.class);
        job.setMapOutputValueClass(Text.class);

        job.setOutputKeyClass(Text.class);
        job.setOutputValueClass(NullWritable.class);

        job.setNumReduceTasks(1);

        FileInputFormat.addInputPath(
                job, new Path(args[0]));

        FileOutputFormat.setOutputPath(
                job, new Path(args[1]));

        System.exit(
                job.waitForCompletion(true) ? 0 : 1);
    }
}
