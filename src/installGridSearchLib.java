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
          
            System.out.println("Installing gridsearch package...");
            WekaPackageManager.installPackageFromURL(urlGrid);
            System.out.println("Installation complete.");

        } catch (Exception e) {
            e.printStackTrace();
        }
    }
}
