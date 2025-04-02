import weka.core.WekaPackageManager;

import java.net.URL;
import java.util.List;
import weka.core.packageManagement.Package;

public class installGridSearchLib {
    public static void main(String[] args) {
        try {
            // List installed packages
            List<Package> installed = WekaPackageManager.getInstalledPackages();
            System.out.println("Installed packages: " + installed.toString());
            URL urlGrid = new URL("\thttp://prdownloads.sourceforge.net/weka/gridSearch1.0.12.zip?download");
            
            // Check if the GridSearch package is installed (the package name is typically "gridsearch")
            if (!installed.contains("gridSearch (1.0.12)")) {
                System.out.println("Installing gridsearch package...");
                WekaPackageManager.installPackageFromURL(urlGrid);
                System.out.println("Installation complete.");
            } else {
                System.out.println("GridSearch package is already installed.");
            }
        } catch (Exception e) {
            e.printStackTrace();
        }
    }
}
