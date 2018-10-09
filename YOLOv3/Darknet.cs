using System;
using System.Collections.Generic;
using System.IO;
using System.Text;
using System.Threading.Tasks;
using System.Text.RegularExpressions;

namespace YOLOv3
{
    public class Darknet
    {
        /// <summary>
        /// Converts a config file into a list of Dictionaries (blocks) which describe the network to be built.
        /// </summary>
        public static List<Dictionary<string, string>> ParseConfigFile(string configFilePath)
        {
            var allBlocks = new List<Dictionary<string, string>>();
            var currentBlock = new Dictionary<string, string>();

            using (StreamReader configFile = File.OpenText(configFilePath))
            {
                string line;
                while((line = configFile.ReadLine()) != null)
                {
                    line = line.Trim();

                    // check if this line is the start of a new block
                    Match newBlockRegex = Regex.Match(line, @"^\[(.+)\]");
                    if (newBlockRegex.Success)
                    {
                        // create new block, and add it to the allBlocks list
                        currentBlock = new Dictionary<string, string>
                        {
                            ["type"] = newBlockRegex.Groups[1].Value
                        };
                        allBlocks.Add(currentBlock);
                    }

                    // check if this line is just a key value pair, and not an empty line or comment line
                    Match keyValueRegex = Regex.Match(line, @"^([^#].+)=(.+)");
                    if (keyValueRegex.Success)
                    {
                        // add key value pair to the current block
                        string key = keyValueRegex.Groups[1].Value.Trim();
                        string value = keyValueRegex.Groups[2].Value.Trim();
                        currentBlock[key] = value;
                    }
                }
            }
            
            return allBlocks;
        }
    }
}
